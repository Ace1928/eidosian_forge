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
def n_at_a_time(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]