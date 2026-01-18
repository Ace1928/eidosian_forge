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
def maybe_stat(conf: Config, path: str) -> Optional[Stat]:
    bucket, blob = split_path(path)
    if blob == '':
        return None
    req = Request(url=build_url('/storage/v1/b/{bucket}/o/{object}', bucket=bucket, object=blob), method='GET', success_codes=(200, 404))
    resp = execute_api_request(conf, req)
    if resp.status != 200:
        return None
    return make_stat(json.loads(resp.data))