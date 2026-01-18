import os
import re
import sys
from codecs import BOM_UTF8, BOM_UTF16, BOM_UTF16_BE, BOM_UTF16_LE
import six
from ._version import __version__
def build_UnaryAdd(self, o):
    return self.build_Const(o.getChildren()[0])