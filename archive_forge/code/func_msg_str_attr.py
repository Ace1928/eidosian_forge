import base64
import collections
import logging
import struct
import functools
from os_ken import exception
from os_ken import utils
from os_ken.lib import stringify
from os_ken.ofproto import ofproto_common
def msg_str_attr(msg_, buf, attr_list=None):
    if attr_list is None:
        attr_list = stringify.obj_attrs(msg_)
    for attr in attr_list:
        val = getattr(msg_, attr, None)
        if val is not None:
            buf += ' %s %s' % (attr, val)
    return buf