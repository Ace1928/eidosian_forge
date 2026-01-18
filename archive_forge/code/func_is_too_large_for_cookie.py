import binascii
import json
from django.conf import settings
from django.contrib.messages.storage.base import BaseStorage, Message
from django.core import signing
from django.http import SimpleCookie
from django.utils.safestring import SafeData, mark_safe
def is_too_large_for_cookie(data):
    return data and len(cookie.value_encode(data)[1]) > self.max_cookie_size