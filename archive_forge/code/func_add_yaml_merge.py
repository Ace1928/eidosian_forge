from __future__ import absolute_import, print_function
import sys
import copy
from ruamel.yaml.compat import ordereddict, PY2, string_types, MutableSliceableSequence
from ruamel.yaml.scalarstring import ScalarString
from ruamel.yaml.anchor import Anchor
def add_yaml_merge(self, value):
    for v in value:
        v[1].add_referent(self)
        for k, v in v[1].items():
            if ordereddict.__contains__(self, k):
                continue
            ordereddict.__setitem__(self, k, v)
    self.merge.extend(value)