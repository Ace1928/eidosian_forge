from __future__ import absolute_import, print_function
import sys
import copy
from ruamel.yaml.compat import ordereddict, PY2, string_types, MutableSliceableSequence
from ruamel.yaml.scalarstring import ScalarString
from ruamel.yaml.anchor import Anchor
@property
def lc(self):
    if not hasattr(self, LineCol.attrib):
        setattr(self, LineCol.attrib, LineCol())
    return getattr(self, LineCol.attrib)