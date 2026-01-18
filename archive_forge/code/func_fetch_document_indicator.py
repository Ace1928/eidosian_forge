from .error import MarkedYAMLError
from .tokens import *
def fetch_document_indicator(self, TokenClass):
    self.unwind_indent(-1)
    self.remove_possible_simple_key()
    self.allow_simple_key = False
    start_mark = self.get_mark()
    self.forward(3)
    end_mark = self.get_mark()
    self.tokens.append(TokenClass(start_mark, end_mark))