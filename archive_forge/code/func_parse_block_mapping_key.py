from .error import MarkedYAMLError
from .tokens import *
from .events import *
from .scanner import *
def parse_block_mapping_key(self):
    if self.check_token(KeyToken):
        token = self.get_token()
        if not self.check_token(KeyToken, ValueToken, BlockEndToken):
            self.states.append(self.parse_block_mapping_value)
            return self.parse_block_node_or_indentless_sequence()
        else:
            self.state = self.parse_block_mapping_value
            return self.process_empty_scalar(token.end_mark)
    if not self.check_token(BlockEndToken):
        token = self.peek_token()
        raise ParserError('while parsing a block mapping', self.marks[-1], 'expected <block end>, but found %r' % token.id, token.start_mark)
    token = self.get_token()
    event = MappingEndEvent(token.start_mark, token.end_mark)
    self.state = self.states.pop()
    self.marks.pop()
    return event