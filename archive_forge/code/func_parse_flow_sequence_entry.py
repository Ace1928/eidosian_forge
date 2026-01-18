from .error import MarkedYAMLError
from .tokens import *
from .events import *
from .scanner import *
def parse_flow_sequence_entry(self, first=False):
    if not self.check_token(FlowSequenceEndToken):
        if not first:
            if self.check_token(FlowEntryToken):
                self.get_token()
            else:
                token = self.peek_token()
                raise ParserError('while parsing a flow sequence', self.marks[-1], "expected ',' or ']', but got %r" % token.id, token.start_mark)
        if self.check_token(KeyToken):
            token = self.peek_token()
            event = MappingStartEvent(None, None, True, token.start_mark, token.end_mark, flow_style=True)
            self.state = self.parse_flow_sequence_entry_mapping_key
            return event
        elif not self.check_token(FlowSequenceEndToken):
            self.states.append(self.parse_flow_sequence_entry)
            return self.parse_flow_node()
    token = self.get_token()
    event = SequenceEndEvent(token.start_mark, token.end_mark)
    self.state = self.states.pop()
    self.marks.pop()
    return event