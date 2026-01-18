from .error import MarkedYAMLError
from .tokens import *
def scan_tag_directive_handle(self, start_mark):
    value = self.scan_tag_handle('directive', start_mark)
    ch = self.peek()
    if ch != ' ':
        raise ScannerError('while scanning a directive', start_mark, "expected ' ', but found %r" % ch, self.get_mark())
    return value