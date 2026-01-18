from __future__ import absolute_import, print_function
import warnings
from ruamel.yaml.error import MarkedYAMLError, ReusedAnchorWarning
from ruamel.yaml.compat import utf8, nprint, nprintf  # NOQA
from ruamel.yaml.events import (
from ruamel.yaml.nodes import MappingNode, ScalarNode, SequenceNode
def compose_document(self):
    self.parser.get_event()
    node = self.compose_node(None, None)
    self.parser.get_event()
    self.anchors = {}
    return node