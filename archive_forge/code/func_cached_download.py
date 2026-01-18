import copy
import fnmatch
import inspect
import io
import json
import os
import re
import shutil
import stat
import tempfile
import time
import uuid
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, BinaryIO, Dict, Generator, Literal, Optional, Tuple, Union
from urllib.parse import quote, urlparse
import requests
from filelock import FileLock
from huggingface_hub import constants
from . import __version__  # noqa: F401 # for backward compatibility
from .constants import (
from .utils import (
from .utils._deprecation import _deprecate_method
from .utils._headers import _http_user_agent
from .utils._runtime import _PY_VERSION  # noqa: F401 # for backward compatibility
from .utils._typing import HTTP_METHOD_T
from .utils.insecure_hashlib import sha256
@validate_hf_hub_args
def cached_download(url: str, *, library_name: Optional[str]=None, library_version: Optional[str]=None, cache_dir: Union[str, Path, None]=None, user_agent: Union[Dict, str, None]=None, force_download: bool=False, force_filename: Optional[str]=None, proxies: Optional[Dict]=None, etag_timeout: float=DEFAULT_ETAG_TIMEOUT, resume_download: bool=False, token: Union[bool, str, None]=None, local_files_only: bool=False, legacy_cache_layout: bool=False) -> str:
    """
    Download from a given URL and cache it if it's not already present in the
    local cache.

    Given a URL, this function looks for the corresponding file in the local
    cache. If it's not there, download it. Then return the path to the cached
    file.

    Will raise errors tailored to the Hugging Face Hub.

    Args:
        url (`str`):
            The path to the file to be downloaded.
        library_name (`str`, *optional*):
            The name of the library to which the object corresponds.
        library_version (`str`, *optional*):
            The version of the library.
        cache_dir (`str`, `Path`, *optional*):
            Path to the folder where cached files are stored.
        user_agent (`dict`, `str`, *optional*):
            The user-agent info in the form of a dictionary or a string.
        force_download (`bool`, *optional*, defaults to `False`):
            Whether the file should be downloaded even if it already exists in
            the local cache.
        force_filename (`str`, *optional*):
            Use this name instead of a generated file name.
        proxies (`dict`, *optional*):
            Dictionary mapping protocol to the URL of the proxy passed to
            `requests.request`.
        etag_timeout (`float`, *optional* defaults to `10`):
            When fetching ETag, how many seconds to wait for the server to send
            data before giving up which is passed to `requests.request`.
        resume_download (`bool`, *optional*, defaults to `False`):
            If `True`, resume a previously interrupted download.
        token (`bool`, `str`, *optional*):
            A token to be used for the download.
                - If `True`, the token is read from the HuggingFace config
                  folder.
                - If a string, it's used as the authentication token.
        local_files_only (`bool`, *optional*, defaults to `False`):
            If `True`, avoid downloading the file and return the path to the
            local cached file if it exists.
        legacy_cache_layout (`bool`, *optional*, defaults to `False`):
            Set this parameter to `True` to mention that you'd like to continue
            the old cache layout. Putting this to `True` manually will not raise
            any warning when using `cached_download`. We recommend using
            `hf_hub_download` to take advantage of the new cache.

    Returns:
        Local path (string) of file or if networking is off, last version of
        file cached on disk.

    <Tip>

    Raises the following errors:

        - [`EnvironmentError`](https://docs.python.org/3/library/exceptions.html#EnvironmentError)
          if `token=True` and the token cannot be found.
        - [`OSError`](https://docs.python.org/3/library/exceptions.html#OSError)
          if ETag cannot be determined.
        - [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
          if some parameter value is invalid
        - [`~utils.RepositoryNotFoundError`]
          If the repository to download from cannot be found. This may be because it doesn't exist,
          or because it is set to `private` and you do not have access.
        - [`~utils.RevisionNotFoundError`]
          If the revision to download from cannot be found.
        - [`~utils.EntryNotFoundError`]
          If the file to download cannot be found.
        - [`~utils.LocalEntryNotFoundError`]
          If network is disabled or unavailable and file is not found in cache.

    </Tip>
    """
    if HF_HUB_ETAG_TIMEOUT != DEFAULT_ETAG_TIMEOUT:
        etag_timeout = HF_HUB_ETAG_TIMEOUT
    if not legacy_cache_layout:
        warnings.warn("'cached_download' is the legacy way to download files from the HF hub, please consider upgrading to 'hf_hub_download'", FutureWarning)
    if cache_dir is None:
        cache_dir = HF_HUB_CACHE
    if isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    headers = build_hf_headers(token=token, library_name=library_name, library_version=library_version, user_agent=user_agent)
    url_to_download = url
    etag = None
    expected_size = None
    if not local_files_only:
        try:
            headers['Accept-Encoding'] = 'identity'
            r = _request_wrapper(method='HEAD', url=url, headers=headers, allow_redirects=False, follow_relative_redirects=True, proxies=proxies, timeout=etag_timeout)
            headers.pop('Accept-Encoding', None)
            hf_raise_for_status(r)
            etag = r.headers.get(HUGGINGFACE_HEADER_X_LINKED_ETAG) or r.headers.get('ETag')
            if etag is None:
                raise FileMetadataError("Distant resource does not have an ETag, we won't be able to reliably ensure reproducibility.")
            expected_size = _int_or_none(r.headers.get('Content-Length'))
            if 300 <= r.status_code <= 399:
                url_to_download = r.headers['Location']
                headers.pop('authorization', None)
                expected_size = None
        except (requests.exceptions.SSLError, requests.exceptions.ProxyError):
            raise
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout, OfflineModeIsEnabled):
            pass
    filename = force_filename if force_filename is not None else url_to_filename(url, etag)
    cache_path = os.path.join(cache_dir, filename)
    if etag is None:
        if os.path.exists(cache_path) and (not force_download):
            return cache_path
        else:
            matching_files = [file for file in fnmatch.filter(os.listdir(cache_dir), filename.split('.')[0] + '.*') if not file.endswith('.json') and (not file.endswith('.lock'))]
            if len(matching_files) > 0 and (not force_download) and (force_filename is None):
                return os.path.join(cache_dir, matching_files[-1])
            elif local_files_only:
                raise LocalEntryNotFoundError("Cannot find the requested files in the cached path and outgoing traffic has been disabled. To enable model look-ups and downloads online, set 'local_files_only' to False.")
            else:
                raise LocalEntryNotFoundError('Connection error, and we cannot find the requested files in the cached path. Please try again or make sure your Internet connection is on.')
    if os.path.exists(cache_path) and (not force_download):
        return cache_path
    lock_path = cache_path + '.lock'
    if os.name == 'nt' and len(os.path.abspath(lock_path)) > 255:
        lock_path = '\\\\?\\' + os.path.abspath(lock_path)
    if os.name == 'nt' and len(os.path.abspath(cache_path)) > 255:
        cache_path = '\\\\?\\' + os.path.abspath(cache_path)
    with FileLock(lock_path):
        if os.path.exists(cache_path) and (not force_download):
            return cache_path
        if resume_download:
            incomplete_path = cache_path + '.incomplete'

            @contextmanager
            def _resumable_file_manager() -> Generator[io.BufferedWriter, None, None]:
                with open(incomplete_path, 'ab') as f:
                    yield f
            temp_file_manager = _resumable_file_manager
            if os.path.exists(incomplete_path):
                resume_size = os.stat(incomplete_path).st_size
            else:
                resume_size = 0
        else:
            temp_file_manager = partial(tempfile.NamedTemporaryFile, mode='wb', dir=cache_dir, delete=False)
            resume_size = 0
        with temp_file_manager() as temp_file:
            logger.info('downloading %s to %s', url, temp_file.name)
            http_get(url_to_download, temp_file, proxies=proxies, resume_size=resume_size, headers=headers, expected_size=expected_size)
        logger.info('storing %s in cache at %s', url, cache_path)
        _chmod_and_replace(temp_file.name, cache_path)
        if force_filename is None:
            logger.info('creating metadata file for %s', cache_path)
            meta = {'url': url, 'etag': etag}
            meta_path = cache_path + '.json'
            with open(meta_path, 'w') as meta_file:
                json.dump(meta, meta_file)
    return cache_path