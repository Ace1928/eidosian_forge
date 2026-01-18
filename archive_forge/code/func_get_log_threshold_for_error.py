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
def get_log_threshold_for_error(conf: Config, err: str) -> int:
    if any((substr in err for substr in COMMON_ERROR_SUBSTRINGS)):
        return conf.retry_common_log_threshold
    else:
        return conf.retry_log_threshold