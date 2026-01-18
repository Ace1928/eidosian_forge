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
def n_groups(seq, n):
    items_per_group = (len(seq) - 1) // n + 1
    return n_at_a_time(seq, items_per_group)