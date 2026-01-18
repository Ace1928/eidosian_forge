import base64
import binascii
import concurrent.futures
import datetime
import hashlib
import json
import math
import os
import platform
import socket
import time
import urllib.parse
from typing import Any, Dict, Iterator, List, Mapping, Optional, Tuple
import urllib3
from blobfile import _common as common
from blobfile._common import (
def list_blobs(conf: Config, path: str, delimiter: Optional[str]=None) -> Iterator[DirEntry]:
    params = {}
    if delimiter is not None:
        params['delimiter'] = delimiter
    bucket, prefix = split_path(path)
    it = _create_page_iterator(conf=conf, url=build_url('/storage/v1/b/{bucket}/o', bucket=bucket), method='GET', params=dict(prefix=prefix, **params))
    for result in it:
        yield from _get_entries(bucket, result)