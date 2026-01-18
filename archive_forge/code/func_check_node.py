from __future__ import absolute_import, print_function
import warnings
from ruamel.yaml.error import MarkedYAMLError, ReusedAnchorWarning
from ruamel.yaml.compat import utf8, nprint, nprintf  # NOQA
from ruamel.yaml.events import (
from ruamel.yaml.nodes import MappingNode, ScalarNode, SequenceNode
def check_node(self):
    if self.parser.check_event(StreamStartEvent):
        self.parser.get_event()
    return not self.parser.check_event(StreamEndEvent)