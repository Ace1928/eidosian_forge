from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from ruamel import yaml
import six
def construct_yaml_int(self, node):
    return self._ScalarObject(node, int(node.value))