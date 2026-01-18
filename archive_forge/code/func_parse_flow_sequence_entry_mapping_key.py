from .error import MarkedYAMLError
from .tokens import *
from .events import *
from .scanner import *
def parse_flow_sequence_entry_mapping_key(self):
    token = self.get_token()
    if not self.check_token(ValueToken, FlowEntryToken, FlowSequenceEndToken):
        self.states.append(self.parse_flow_sequence_entry_mapping_value)
        return self.parse_flow_node()
    else:
        self.state = self.parse_flow_sequence_entry_mapping_value
        return self.process_empty_scalar(token.end_mark)