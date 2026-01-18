import hmac
import json
import base64
import datetime
from hashlib import sha256
from libcloud.dns.base import Zone, Record, DNSDriver
from libcloud.dns.types import (
from libcloud.utils.py3 import b, httplib
from libcloud.common.base import JsonResponse, ConnectionUserAndKey
from libcloud.common.types import LibcloudError, ProviderError, InvalidCredsError
def calculate_auth_signature(self, secret_key, method, url, timestamp):
    b64_hmac = base64.b64encode(hmac.new(b(secret_key), b(method) + b(url) + b(timestamp), digestmod=sha256).digest())
    return b64_hmac.decode('utf-8')