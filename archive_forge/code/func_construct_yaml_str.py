from __future__ import print_function, absolute_import, division
import datetime
import base64
import binascii
import re
import sys
import types
import warnings
from ruamel.yaml.error import (MarkedYAMLError, MarkedYAMLFutureWarning,
from ruamel.yaml.nodes import *                               # NOQA
from ruamel.yaml.nodes import (SequenceNode, MappingNode, ScalarNode)
from ruamel.yaml.compat import (utf8, builtins_module, to_str, PY2, PY3,  # NOQA
from ruamel.yaml.comments import *                               # NOQA
from ruamel.yaml.comments import (CommentedMap, CommentedOrderedMap, CommentedSet,
from ruamel.yaml.scalarstring import (SingleQuotedScalarString, DoubleQuotedScalarString,
from ruamel.yaml.scalarint import ScalarInt, BinaryInt, OctalInt, HexInt, HexCapsInt
from ruamel.yaml.scalarfloat import ScalarFloat
from ruamel.yaml.scalarbool import ScalarBoolean
from ruamel.yaml.timestamp import TimeStamp
from ruamel.yaml.util import RegExp
def construct_yaml_str(self, node):
    value = self.construct_scalar(node)
    if isinstance(value, ScalarString):
        return value
    if PY3:
        return value
    try:
        return value.encode('ascii')
    except AttributeError:
        return value
    except UnicodeEncodeError:
        return value