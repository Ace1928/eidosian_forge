from __future__ import print_function, absolute_import, division
import datetime
import base64
import binascii
import re
import sys
import types
import warnings
from .error import (MarkedYAMLError, MarkedYAMLFutureWarning,
from .nodes import *                               # NOQA
from .nodes import (SequenceNode, MappingNode, ScalarNode)
from .compat import (utf8, builtins_module, to_str, PY2, PY3,  # NOQA
from .compat import ordereddict, Hashable, MutableSequence  # type: ignore
from .compat import MutableMapping  # type: ignore
from .comments import *                               # NOQA
from .comments import (CommentedMap, CommentedOrderedMap, CommentedSet,
from .scalarstring import (SingleQuotedScalarString, DoubleQuotedScalarString,
from .scalarint import ScalarInt, BinaryInt, OctalInt, HexInt, HexCapsInt
from .scalarfloat import ScalarFloat
from .scalarbool import ScalarBoolean
from .timestamp import TimeStamp
from .util import RegExp
def construct_non_recursive_object(self, node, tag=None):
    constructor = None
    tag_suffix = None
    if tag is None:
        tag = node.tag
    if tag in self.yaml_constructors:
        constructor = self.yaml_constructors[tag]
    else:
        for tag_prefix in self.yaml_multi_constructors:
            if tag.startswith(tag_prefix):
                tag_suffix = tag[len(tag_prefix):]
                constructor = self.yaml_multi_constructors[tag_prefix]
                break
        else:
            if None in self.yaml_multi_constructors:
                tag_suffix = tag
                constructor = self.yaml_multi_constructors[None]
            elif None in self.yaml_constructors:
                constructor = self.yaml_constructors[None]
            elif isinstance(node, ScalarNode):
                constructor = self.__class__.construct_scalar
            elif isinstance(node, SequenceNode):
                constructor = self.__class__.construct_sequence
            elif isinstance(node, MappingNode):
                constructor = self.__class__.construct_mapping
    if tag_suffix is None:
        data = constructor(self, node)
    else:
        data = constructor(self, tag_suffix, node)
    if isinstance(data, types.GeneratorType):
        generator = data
        data = next(generator)
        if self.deep_construct:
            for _dummy in generator:
                pass
        else:
            self.state_generators.append(generator)
    return data