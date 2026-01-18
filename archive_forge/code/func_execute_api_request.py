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
def execute_api_request(conf: Config, req: Request) -> 'urllib3.BaseHTTPResponse':

    def build_req() -> Request:
        return create_api_request(req, auth=access_token_manager.get_token(conf, key=''))
    return common.execute_request(conf, build_req)