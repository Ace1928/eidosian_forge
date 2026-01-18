from .error import MarkedYAMLError
from .tokens import *
from .events import *
from .scanner import *
def parse_block_sequence_entry(self):
    if self.check_token(BlockEntryToken):
        token = self.get_token()
        if not self.check_token(BlockEntryToken, BlockEndToken):
            self.states.append(self.parse_block_sequence_entry)
            return self.parse_block_node()
        else:
            self.state = self.parse_block_sequence_entry
            return self.process_empty_scalar(token.end_mark)
    if not self.check_token(BlockEndToken):
        token = self.peek_token()
        raise ParserError('while parsing a block collection', self.marks[-1], 'expected <block end>, but found %r' % token.id, token.start_mark)
    token = self.get_token()
    event = SequenceEndEvent(token.start_mark, token.end_mark)
    self.state = self.states.pop()
    self.marks.pop()
    return event