from collections import abc
import decimal
import random
import weakref
from oslo_log import log as logging
from oslo_utils import timeutils
from oslo_utils import uuidutils
from neutron_lib._i18n import _
def safe_decode_utf8(s):
    """Safe decode a str from UTF.

    :param s: The str to decode.
    :returns: The decoded str.
    """
    if isinstance(s, bytes):
        return s.decode('utf-8', 'surrogateescape')
    return s