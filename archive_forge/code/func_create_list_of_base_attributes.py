import base64
import collections
import logging
import struct
import functools
from os_ken import exception
from os_ken import utils
from os_ken.lib import stringify
from os_ken.ofproto import ofproto_common
def create_list_of_base_attributes(f):

    @functools.wraps(f)
    def wrapper(self, *args, **kwargs):
        ret = f(self, *args, **kwargs)
        cls = self.__class__
        if '_base_attributes' not in cls.__dict__:
            cls._base_attributes = set(dir(self))
        return ret
    return wrapper