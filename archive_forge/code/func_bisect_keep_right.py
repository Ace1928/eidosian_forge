import binascii
import json
from django.conf import settings
from django.contrib.messages.storage.base import BaseStorage, Message
from django.core import signing
from django.http import SimpleCookie
from django.utils.safestring import SafeData, mark_safe
def bisect_keep_right(a, fn):
    """
    Find the index of the first element from the end of the array that verifies
    the given condition.
    The function is applied from the pivot to the end of array.
    """
    lo = 0
    hi = len(a)
    while lo < hi:
        mid = (lo + hi) // 2
        if fn(a[mid:]):
            lo = mid + 1
        else:
            hi = mid
    return lo