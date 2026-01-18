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
def build_req_list_keys() -> Request:
    req = Request(method='POST', url=f'https://management.azure.com{storage_account_id}/listKeys', params={'api-version': '2019-04-01'})
    return create_api_request(req, auth=auth)