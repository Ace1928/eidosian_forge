from .error import MarkedYAMLError
from .tokens import *
from .events import *
from .scanner import *
def parse_stream_start(self):
    token = self.get_token()
    event = StreamStartEvent(token.start_mark, token.end_mark, encoding=token.encoding)
    self.state = self.parse_implicit_document_start
    return event