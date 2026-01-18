from __future__ import annotations
import base64
import hashlib
import json
import os
import tempfile
import uuid
from typing import Optional
import requests
def read_file_cached(blobpath: str, expected_hash: Optional[str]=None) -> bytes:
    user_specified_cache = True
    if 'TIKTOKEN_CACHE_DIR' in os.environ:
        cache_dir = os.environ['TIKTOKEN_CACHE_DIR']
    elif 'DATA_GYM_CACHE_DIR' in os.environ:
        cache_dir = os.environ['DATA_GYM_CACHE_DIR']
    else:
        cache_dir = os.path.join(tempfile.gettempdir(), 'data-gym-cache')
        user_specified_cache = False
    if cache_dir == '':
        return read_file(blobpath)
    cache_key = hashlib.sha1(blobpath.encode()).hexdigest()
    cache_path = os.path.join(cache_dir, cache_key)
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            data = f.read()
        if expected_hash is None or check_hash(data, expected_hash):
            return data
        try:
            os.remove(cache_path)
        except OSError:
            pass
    contents = read_file(blobpath)
    if expected_hash and (not check_hash(contents, expected_hash)):
        raise ValueError(f'Hash mismatch for data downloaded from {blobpath} (expected {expected_hash}). This may indicate a corrupted download. Please try again.')
    try:
        os.makedirs(cache_dir, exist_ok=True)
        tmp_filename = cache_path + '.' + str(uuid.uuid4()) + '.tmp'
        with open(tmp_filename, 'wb') as f:
            f.write(contents)
        os.rename(tmp_filename, cache_path)
    except OSError:
        if user_specified_cache:
            raise
    return contents