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
def build_req() -> Request:
    return Request(method='GET', url='http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token', headers={'Metadata-Flavor': 'Google'})