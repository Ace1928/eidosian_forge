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
def config_true_value(value):
    """
    Returns True if the value is either True or a string in TRUE_VALUES.
    Returns False otherwise.
    This function comes from swift.common.utils.config_true_value()
    """
    return value is True or (isinstance(value, str) and value.lower() in TRUE_VALUES)