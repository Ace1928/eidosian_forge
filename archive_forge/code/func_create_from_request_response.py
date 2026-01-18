import hashlib
import io
import json
import os
import platform
import random
import socket
import ssl
import threading
import time
import urllib.parse
from typing import (
import filelock
import urllib3
from blobfile import _xml as xml
@classmethod
def create_from_request_response(cls, message: str, request: Request, response: 'urllib3.BaseHTTPResponse') -> Any:
    err, err_desc, err_headers = _extract_error_from_response(response)
    return cls(message=message, request_string=str(request), response_status=response.status, error=err, error_description=err_desc, error_headers=err_headers)