from .error import MarkedYAMLError
from .tokens import *
from .events import *
from .scanner import *
def parse_implicit_document_start(self):
    if not self.check_token(DirectiveToken, DocumentStartToken, StreamEndToken):
        self.tag_handles = self.DEFAULT_TAGS
        token = self.peek_token()
        start_mark = end_mark = token.start_mark
        event = DocumentStartEvent(start_mark, end_mark, explicit=False)
        self.states.append(self.parse_document_end)
        self.state = self.parse_block_node
        return event
    else:
        return self.parse_document_start()