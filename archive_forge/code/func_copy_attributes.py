from __future__ import absolute_import, print_function
import sys
import copy
from ruamel.yaml.compat import ordereddict, PY2, string_types, MutableSliceableSequence
from ruamel.yaml.scalarstring import ScalarString
from ruamel.yaml.anchor import Anchor
def copy_attributes(self, t, deep=False):
    for a in [Comment.attrib, Format.attrib, LineCol.attrib, Anchor.attrib, Tag.attrib, merge_attrib]:
        if hasattr(self, a):
            if deep:
                setattr(t, a, copy.deepcopy(getattr(self, a)))
            else:
                setattr(t, a, getattr(self, a))