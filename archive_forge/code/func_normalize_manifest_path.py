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
def normalize_manifest_path(path):
    if path.startswith('/'):
        return path[1:]
    return path