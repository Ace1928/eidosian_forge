from .error import MarkedYAMLError
from .tokens import *
from .events import *
from .scanner import *
def peek_event(self):
    if self.current_event is None:
        if self.state:
            self.current_event = self.state()
    return self.current_event