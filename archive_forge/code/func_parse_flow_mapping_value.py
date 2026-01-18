from .error import MarkedYAMLError
from .tokens import *
from .events import *
from .scanner import *
def parse_flow_mapping_value(self):
    if self.check_token(ValueToken):
        token = self.get_token()
        if not self.check_token(FlowEntryToken, FlowMappingEndToken):
            self.states.append(self.parse_flow_mapping_key)
            return self.parse_flow_node()
        else:
            self.state = self.parse_flow_mapping_key
            return self.process_empty_scalar(token.end_mark)
    else:
        self.state = self.parse_flow_mapping_key
        token = self.peek_token()
        return self.process_empty_scalar(token.start_mark)