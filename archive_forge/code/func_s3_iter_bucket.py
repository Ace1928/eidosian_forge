import logging
from smart_open import version  # noqa: E402
from .smart_open_lib import open, parse_uri, smart_open, register_compressor  # noqa: E402
def s3_iter_bucket(bucket_name, prefix='', accept_key=None, key_limit=None, workers=16, retries=3, **session_kwargs):
    """Deprecated.  Use smart_open.s3.iter_bucket instead."""
    global _WARNED
    from .s3 import iter_bucket
    if not _WARNED:
        logger.warning(_WARNING)
        _WARNED = True
    return iter_bucket(bucket_name=bucket_name, prefix=prefix, accept_key=accept_key, key_limit=key_limit, workers=workers, retries=retries, session_kwargs=session_kwargs)