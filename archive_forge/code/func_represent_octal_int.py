from __future__ import print_function, absolute_import, division
from ruamel.yaml.error import *  # NOQA
from ruamel.yaml.nodes import *  # NOQA
from ruamel.yaml.compat import text_type, binary_type, to_unicode, PY2, PY3, ordereddict
from ruamel.yaml.compat import nprint, nprintf  # NOQA
from ruamel.yaml.scalarstring import (
from ruamel.yaml.scalarint import ScalarInt, BinaryInt, OctalInt, HexInt, HexCapsInt
from ruamel.yaml.scalarfloat import ScalarFloat
from ruamel.yaml.scalarbool import ScalarBoolean
from ruamel.yaml.timestamp import TimeStamp
import datetime
import sys
import types
from ruamel.yaml.comments import (
def represent_octal_int(self, data):
    if data._width is not None:
        s = '{:0{}o}'.format(data, data._width)
    else:
        s = format(data, 'o')
    anchor = data.yaml_anchor(any=True)
    return self.insert_underscore('0o', s, data._underscore, anchor=anchor)