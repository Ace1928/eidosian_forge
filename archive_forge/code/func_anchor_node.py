from __future__ import absolute_import
from ruamel.yaml.error import YAMLError
from ruamel.yaml.compat import nprint, DBG_NODE, dbg, string_types, nprintf  # NOQA
from ruamel.yaml.util import RegExp
from ruamel.yaml.events import (
from ruamel.yaml.nodes import MappingNode, ScalarNode, SequenceNode
def anchor_node(self, node):
    if node in self.anchors:
        if self.anchors[node] is None:
            self.anchors[node] = self.generate_anchor(node)
    else:
        anchor = None
        try:
            if node.anchor.always_dump:
                anchor = node.anchor.value
        except:
            pass
        self.anchors[node] = anchor
        if isinstance(node, SequenceNode):
            for item in node.value:
                self.anchor_node(item)
        elif isinstance(node, MappingNode):
            for key, value in node.value:
                self.anchor_node(key)
                self.anchor_node(value)