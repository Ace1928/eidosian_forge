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
def maybe_update_md5(conf: Config, path: str, generation: str, hexdigest: str) -> bool:
    bucket, blob = split_path(path)
    req = Request(url=build_url('/storage/v1/b/{bucket}/o/{object}', bucket=bucket, object=blob), method='PATCH', params=dict(ifGenerationMatch=generation), data=dict(metadata={'md5': hexdigest}), success_codes=(200, 404, 412))
    resp = execute_api_request(conf, req)
    return resp.status == 200