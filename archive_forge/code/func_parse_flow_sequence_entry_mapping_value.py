from .error import MarkedYAMLError
from .tokens import *
from .events import *
from .scanner import *
def parse_flow_sequence_entry_mapping_value(self):
    if self.check_token(ValueToken):
        token = self.get_token()
        if not self.check_token(FlowEntryToken, FlowSequenceEndToken):
            self.states.append(self.parse_flow_sequence_entry_mapping_end)
            return self.parse_flow_node()
        else:
            self.state = self.parse_flow_sequence_entry_mapping_end
            return self.process_empty_scalar(token.end_mark)
    else:
        self.state = self.parse_flow_sequence_entry_mapping_end
        token = self.peek_token()
        return self.process_empty_scalar(token.start_mark)