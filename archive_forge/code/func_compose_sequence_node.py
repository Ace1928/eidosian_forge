from __future__ import absolute_import, print_function
import warnings
from ruamel.yaml.error import MarkedYAMLError, ReusedAnchorWarning
from ruamel.yaml.compat import utf8, nprint, nprintf  # NOQA
from ruamel.yaml.events import (
from ruamel.yaml.nodes import MappingNode, ScalarNode, SequenceNode
def compose_sequence_node(self, anchor):
    start_event = self.parser.get_event()
    tag = start_event.tag
    if tag is None or tag == u'!':
        tag = self.resolver.resolve(SequenceNode, None, start_event.implicit)
    node = SequenceNode(tag, [], start_event.start_mark, None, flow_style=start_event.flow_style, comment=start_event.comment, anchor=anchor)
    if anchor is not None:
        self.anchors[anchor] = node
    index = 0
    while not self.parser.check_event(SequenceEndEvent):
        node.value.append(self.compose_node(node, index))
        index += 1
    end_event = self.parser.get_event()
    if node.flow_style is True and end_event.comment is not None:
        if node.comment is not None:
            nprint('Warning: unexpected end_event commment in sequence node {}'.format(node.flow_style))
        node.comment = end_event.comment
    node.end_mark = end_event.end_mark
    self.check_end_doc_comment(end_event, node)
    return node