from .error import MarkedYAMLError
from .tokens import *
def scan_tag_directive_value(self, start_mark):
    while self.peek() == ' ':
        self.forward()
    handle = self.scan_tag_directive_handle(start_mark)
    while self.peek() == ' ':
        self.forward()
    prefix = self.scan_tag_directive_prefix(start_mark)
    return (handle, prefix)