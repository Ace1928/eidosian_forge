from .error import MarkedYAMLError
from .tokens import *
from .events import *
from .scanner import *
def parse_document_start(self):
    while self.check_token(DocumentEndToken):
        self.get_token()
    if not self.check_token(StreamEndToken):
        token = self.peek_token()
        start_mark = token.start_mark
        version, tags = self.process_directives()
        if not self.check_token(DocumentStartToken):
            raise ParserError(None, None, "expected '<document start>', but found %r" % self.peek_token().id, self.peek_token().start_mark)
        token = self.get_token()
        end_mark = token.end_mark
        event = DocumentStartEvent(start_mark, end_mark, explicit=True, version=version, tags=tags)
        self.states.append(self.parse_document_end)
        self.state = self.parse_document_content
    else:
        token = self.get_token()
        event = StreamEndEvent(token.start_mark, token.end_mark)
        assert not self.states
        assert not self.marks
        self.state = None
    return event