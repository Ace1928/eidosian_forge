from .error import MarkedYAMLError
from .tokens import *
def fetch_flow_collection_end(self, TokenClass):
    self.remove_possible_simple_key()
    self.flow_level -= 1
    self.allow_simple_key = False
    start_mark = self.get_mark()
    self.forward()
    end_mark = self.get_mark()
    self.tokens.append(TokenClass(start_mark, end_mark))