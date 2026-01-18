from __future__ import absolute_import, print_function
import warnings
from ruamel.yaml.error import MarkedYAMLError, ReusedAnchorWarning
from ruamel.yaml.compat import utf8, nprint, nprintf  # NOQA
from ruamel.yaml.events import (
from ruamel.yaml.nodes import MappingNode, ScalarNode, SequenceNode
def compose_node(self, parent, index):
    if self.parser.check_event(AliasEvent):
        event = self.parser.get_event()
        alias = event.anchor
        if alias not in self.anchors:
            raise ComposerError(None, None, 'found undefined alias %r' % utf8(alias), event.start_mark)
        return self.anchors[alias]
    event = self.parser.peek_event()
    anchor = event.anchor
    if anchor is not None:
        if anchor in self.anchors:
            ws = '\nfound duplicate anchor {!r}\nfirst occurrence {}\nsecond occurrence {}'.format(anchor, self.anchors[anchor].start_mark, event.start_mark)
            warnings.warn(ws, ReusedAnchorWarning)
    self.resolver.descend_resolver(parent, index)
    if self.parser.check_event(ScalarEvent):
        node = self.compose_scalar_node(anchor)
    elif self.parser.check_event(SequenceStartEvent):
        node = self.compose_sequence_node(anchor)
    elif self.parser.check_event(MappingStartEvent):
        node = self.compose_mapping_node(anchor)
    self.resolver.ascend_resolver()
    return node