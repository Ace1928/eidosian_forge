import base64
import datetime
import decimal
import inspect
import io
import logging
import netaddr
import re
import sys
import uuid
import weakref
from wsme import exc
def iswsattr(attr):
    if inspect.isfunction(attr) or inspect.ismethod(attr):
        return False
    if isinstance(attr, property) and (not isinstance(attr, wsproperty)):
        return False
    return True