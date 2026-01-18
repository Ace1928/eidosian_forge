import os
import time
import contextlib
from pathlib import Path
import shlex
import shutil
from .hashes import hash_matches, file_hash
from .utils import (
from .downloaders import DOIDownloader, choose_downloader, doi_to_repository
def stream_download(url, fname, known_hash, downloader, pooch=None, retry_if_failed=0):
    """
    Stream the file and check that its hash matches the known one.

    The file is first downloaded to a temporary file name in the cache folder.
    It will be moved to the desired file name only if the hash matches the
    known hash. Otherwise, the temporary file is deleted.

    If the download fails for either a bad connection or a hash mismatch, we
    will retry the download the specified number of times in case the failure
    was due to a network error.
    """
    import requests.exceptions
    if not fname.parent.exists():
        os.makedirs(str(fname.parent))
    download_attempts = 1 + retry_if_failed
    max_wait = 10
    for i in range(download_attempts):
        try:
            with temporary_file(path=str(fname.parent)) as tmp:
                downloader(url, tmp, pooch)
                hash_matches(tmp, known_hash, strict=True, source=str(fname.name))
                shutil.move(tmp, str(fname))
            break
        except (ValueError, requests.exceptions.RequestException):
            if i == download_attempts - 1:
                raise
            retries_left = download_attempts - (i + 1)
            get_logger().info("Failed to download '%s'. Will attempt the download again %d more time%s.", str(fname.name), retries_left, 's' if retries_left > 1 else '')
            time.sleep(min(i + 1, max_wait))