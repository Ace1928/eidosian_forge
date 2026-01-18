from .error import MarkedYAMLError
from .tokens import *
from .events import *
from .scanner import *
def parse_flow_sequence_entry_mapping_end(self):
    self.state = self.parse_flow_sequence_entry
    token = self.peek_token()
    return MappingEndEvent(token.start_mark, token.start_mark)