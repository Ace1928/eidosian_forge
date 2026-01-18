import base64
import binascii
import calendar
import concurrent.futures
import datetime
import hashlib
import hmac
import json
import math
import os
import re
import time
import urllib.parse
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple
import urllib3
from blobfile import _common as common
from blobfile import _xml as xml
from blobfile._common import (
def parallel_remote_copy(conf: Config, executor: concurrent.futures.Executor, src: str, dst: str, return_md5: bool, dst_version: Optional[str]) -> Optional[str]:
    st = maybe_stat(conf, src)
    if st is None:
        raise FileNotFoundError(f"The system cannot find the path specified: '{src}'")
    md5_digest = None
    if st.md5 is not None:
        md5_digest = binascii.unhexlify(st.md5)
    upload_id = rng.randint(0, 2 ** 47 - 1)
    block_ids = []
    min_block_size = st.size // BLOCK_COUNT_LIMIT
    assert min_block_size <= MAX_BLOCK_SIZE
    part_size = max(conf.azure_write_chunk_size, min_block_size)
    i = 0
    start = 0
    futures = []
    while start < st.size:
        block_id = _block_index_to_block_id(i, upload_id)
        future = executor.submit(_put_block_from_url, conf, src, start, min(part_size, st.size - start), dst, block_id)
        futures.append(future)
        block_ids.append(block_id)
        i += 1
        start += part_size
    for future in futures:
        future.result()
    dst_account, dst_container, dst_blob = split_path(dst)
    dst_url = build_url(dst_account, '/{container}/{blob}', container=dst_container, blob=dst_blob)
    _finalize_blob(conf=conf, path=dst, url=dst_url, block_ids=block_ids, md5_digest=md5_digest, version=dst_version)
    return binascii.hexlify(md5_digest).decode('utf8') if return_md5 and md5_digest is not None else None