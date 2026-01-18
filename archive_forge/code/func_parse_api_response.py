import base64
from calendar import timegm
from collections.abc import Mapping
import gzip
import hashlib
import hmac
import io
import json
import logging
import time
import traceback
def parse_api_response(headers, body):
    body = get_body(headers, body)
    charset = 'utf-8'
    content_type = headers.get('content-type', '')
    if '; charset=' in content_type:
        charset = content_type.split('; charset=', 1)[1].split(';', 1)[0]
    return json.loads(body.decode(charset))