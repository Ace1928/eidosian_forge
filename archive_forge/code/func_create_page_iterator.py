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
def create_page_iterator(conf: Config, url: str, method: str, data: Optional[Mapping[str, str]]=None, params: Optional[Mapping[str, str]]=None) -> Iterator[Dict[str, Any]]:
    if params is None:
        p = {}
    else:
        p = dict(params).copy()
    if data is None:
        d = None
    else:
        d = dict(data).copy()
    while True:
        req = Request(url=url, method=method, params=p, data=d, success_codes=(200, 404, INVALID_HOSTNAME_STATUS))
        resp = execute_api_request(conf, req)
        if resp.status in (404, INVALID_HOSTNAME_STATUS):
            return
        result = xml.parse(resp.data, repeated_tags={'BlobPrefix', 'Blob'})['EnumerationResults']
        yield result
        if result['NextMarker'] is None:
            break
        p['marker'] = result['NextMarker']