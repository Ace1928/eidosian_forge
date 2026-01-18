from __future__ import annotations
import datetime
from datetime import timedelta as TimeDelta
import binascii
import sys
import types
import warnings
from collections.abc import Hashable, MutableSequence, MutableMapping
from ruamel.yaml.error import (MarkedYAMLError, MarkedYAMLFutureWarning,
from ruamel.yaml.nodes import *                               # NOQA
from ruamel.yaml.nodes import (SequenceNode, MappingNode, ScalarNode)
from ruamel.yaml.compat import (builtins_module, # NOQA
from ruamel.yaml.compat import ordereddict
from ruamel.yaml.tag import Tag
from ruamel.yaml.comments import *                               # NOQA
from ruamel.yaml.comments import (CommentedMap, CommentedOrderedMap, CommentedSet,
from ruamel.yaml.scalarstring import (SingleQuotedScalarString, DoubleQuotedScalarString,
from ruamel.yaml.scalarint import ScalarInt, BinaryInt, OctalInt, HexInt, HexCapsInt
from ruamel.yaml.scalarfloat import ScalarFloat
from ruamel.yaml.scalarbool import ScalarBoolean
from ruamel.yaml.timestamp import TimeStamp
from ruamel.yaml.util import timestamp_regexp, create_timestamp
def construct_yaml_sbool(self, node: Any) -> Union[bool, ScalarBoolean]:
    b = SafeConstructor.construct_yaml_bool(self, node)
    if node.anchor:
        return ScalarBoolean(b, anchor=node.anchor)
    return b