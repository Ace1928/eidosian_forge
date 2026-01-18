from .error import MarkedYAMLError
from .tokens import *
from .events import *
from .scanner import *
def parse_node(self, block=False, indentless_sequence=False):
    if self.check_token(AliasToken):
        token = self.get_token()
        event = AliasEvent(token.value, token.start_mark, token.end_mark)
        self.state = self.states.pop()
    else:
        anchor = None
        tag = None
        start_mark = end_mark = tag_mark = None
        if self.check_token(AnchorToken):
            token = self.get_token()
            start_mark = token.start_mark
            end_mark = token.end_mark
            anchor = token.value
            if self.check_token(TagToken):
                token = self.get_token()
                tag_mark = token.start_mark
                end_mark = token.end_mark
                tag = token.value
        elif self.check_token(TagToken):
            token = self.get_token()
            start_mark = tag_mark = token.start_mark
            end_mark = token.end_mark
            tag = token.value
            if self.check_token(AnchorToken):
                token = self.get_token()
                end_mark = token.end_mark
                anchor = token.value
        if tag is not None:
            handle, suffix = tag
            if handle is not None:
                if handle not in self.tag_handles:
                    raise ParserError('while parsing a node', start_mark, 'found undefined tag handle %r' % handle, tag_mark)
                tag = self.tag_handles[handle] + suffix
            else:
                tag = suffix
        if start_mark is None:
            start_mark = end_mark = self.peek_token().start_mark
        event = None
        implicit = tag is None or tag == '!'
        if indentless_sequence and self.check_token(BlockEntryToken):
            end_mark = self.peek_token().end_mark
            event = SequenceStartEvent(anchor, tag, implicit, start_mark, end_mark)
            self.state = self.parse_indentless_sequence_entry
        elif self.check_token(ScalarToken):
            token = self.get_token()
            end_mark = token.end_mark
            if token.plain and tag is None or tag == '!':
                implicit = (True, False)
            elif tag is None:
                implicit = (False, True)
            else:
                implicit = (False, False)
            event = ScalarEvent(anchor, tag, implicit, token.value, start_mark, end_mark, style=token.style)
            self.state = self.states.pop()
        elif self.check_token(FlowSequenceStartToken):
            end_mark = self.peek_token().end_mark
            event = SequenceStartEvent(anchor, tag, implicit, start_mark, end_mark, flow_style=True)
            self.state = self.parse_flow_sequence_first_entry
        elif self.check_token(FlowMappingStartToken):
            end_mark = self.peek_token().end_mark
            event = MappingStartEvent(anchor, tag, implicit, start_mark, end_mark, flow_style=True)
            self.state = self.parse_flow_mapping_first_key
        elif block and self.check_token(BlockSequenceStartToken):
            end_mark = self.peek_token().start_mark
            event = SequenceStartEvent(anchor, tag, implicit, start_mark, end_mark, flow_style=False)
            self.state = self.parse_block_sequence_first_entry
        elif block and self.check_token(BlockMappingStartToken):
            end_mark = self.peek_token().start_mark
            event = MappingStartEvent(anchor, tag, implicit, start_mark, end_mark, flow_style=False)
            self.state = self.parse_block_mapping_first_key
        elif anchor is not None or tag is not None:
            event = ScalarEvent(anchor, tag, (implicit, False), '', start_mark, end_mark)
            self.state = self.states.pop()
        else:
            if block:
                node = 'block'
            else:
                node = 'flow'
            token = self.peek_token()
            raise ParserError('while parsing a %s node' % node, start_mark, 'expected the node content, but found %r' % token.id, token.start_mark)
    return event