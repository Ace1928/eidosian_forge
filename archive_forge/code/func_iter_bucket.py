import io
import functools
import logging
import time
import warnings
import smart_open.bytebuffer
import smart_open.concurrency
import smart_open.utils
from smart_open import constants
def iter_bucket(bucket_name, prefix='', accept_key=None, key_limit=None, workers=16, retries=3, **session_kwargs):
    """
    Iterate and download all S3 objects under `s3://bucket_name/prefix`.

    Parameters
    ----------
    bucket_name: str
        The name of the bucket.
    prefix: str, optional
        Limits the iteration to keys starting with the prefix.
    accept_key: callable, optional
        This is a function that accepts a key name (unicode string) and
        returns True/False, signalling whether the given key should be downloaded.
        The default behavior is to accept all keys.
    key_limit: int, optional
        If specified, the iterator will stop after yielding this many results.
    workers: int, optional
        The number of subprocesses to use.
    retries: int, optional
        The number of time to retry a failed download.
    session_kwargs: dict, optional
        Keyword arguments to pass when creating a new session.
        For a list of available names and values, see:
        https://boto3.amazonaws.com/v1/documentation/api/latest/reference/core/session.html#boto3.session.Session


    Yields
    ------
    str
        The full key name (does not include the bucket name).
    bytes
        The full contents of the key.

    Notes
    -----
    The keys are processed in parallel, using `workers` processes (default: 16),
    to speed up downloads greatly. If multiprocessing is not available, thus
    _MULTIPROCESSING is False, this parameter will be ignored.

    Examples
    --------

      >>> # get all JSON files under "mybucket/foo/"
      >>> for key, content in iter_bucket(
      ...         bucket_name, prefix='foo/',
      ...         accept_key=lambda key: key.endswith('.json')):
      ...     print key, len(content)

      >>> # limit to 10k files, using 32 parallel workers (default is 16)
      >>> for key, content in iter_bucket(bucket_name, key_limit=10000, workers=32):
      ...     print key, len(content)
    """
    if accept_key is None:
        accept_key = _accept_all
    try:
        bucket_name = bucket_name.name
    except AttributeError:
        pass
    total_size, key_no = (0, -1)
    key_iterator = _list_bucket(bucket_name, prefix=prefix, accept_key=accept_key, **session_kwargs)
    download_key = functools.partial(_download_key, bucket_name=bucket_name, retries=retries, **session_kwargs)
    with smart_open.concurrency.create_pool(processes=workers) as pool:
        result_iterator = pool.imap_unordered(download_key, key_iterator)
        key_no = 0
        while True:
            try:
                key, content = result_iterator.__next__()
                if key_no % 1000 == 0:
                    logger.info('yielding key #%i: %s, size %i (total %.1fMB)', key_no, key, len(content), total_size / 1024.0 ** 2)
                yield (key, content)
                total_size += len(content)
                if key_limit is not None and key_no + 1 >= key_limit:
                    break
            except botocore.exceptions.ClientError as err:
                if not ('Error' in err.response and err.response['Error'].get('Code') == '404'):
                    raise err
            except StopIteration:
                break
            key_no += 1
    logger.info('processed %i keys, total size %i' % (key_no + 1, total_size))