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
def represent_key(self, data):
    if isinstance(data, CommentedKeySeq):
        self.alias_key = None
        return self.represent_sequence(u'tag:yaml.org,2002:seq', data, flow_style=True)
    if isinstance(data, CommentedKeyMap):
        self.alias_key = None
        return self.represent_mapping(u'tag:yaml.org,2002:map', data, flow_style=True)
    return SafeRepresenter.represent_key(self, data)