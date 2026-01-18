from __future__ import unicode_literals
from past.builtins import basestring
from .dag import KwargReprNode
from ._utils import escape_chars, get_hash_int
from builtins import object
import os
def filter_operator(name=None):
    return stream_operator(stream_classes={FilterableStream}, name=name)