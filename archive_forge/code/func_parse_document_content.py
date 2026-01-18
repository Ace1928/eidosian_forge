from .error import MarkedYAMLError
from .tokens import *
from .events import *
from .scanner import *
def parse_document_content(self):
    if self.check_token(DirectiveToken, DocumentStartToken, DocumentEndToken, StreamEndToken):
        event = self.process_empty_scalar(self.peek_token().start_mark)
        self.state = self.states.pop()
        return event
    else:
        return self.parse_block_node()