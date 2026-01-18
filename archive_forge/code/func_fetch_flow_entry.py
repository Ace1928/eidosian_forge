from .error import MarkedYAMLError
from .tokens import *
def fetch_flow_entry(self):
    self.allow_simple_key = True
    self.remove_possible_simple_key()
    start_mark = self.get_mark()
    self.forward()
    end_mark = self.get_mark()
    self.tokens.append(FlowEntryToken(start_mark, end_mark))