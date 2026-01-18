from .error import MarkedYAMLError
from .tokens import *
def scan_block_scalar_indicators(self, start_mark):
    chomping = None
    increment = None
    ch = self.peek()
    if ch in '+-':
        if ch == '+':
            chomping = True
        else:
            chomping = False
        self.forward()
        ch = self.peek()
        if ch in '0123456789':
            increment = int(ch)
            if increment == 0:
                raise ScannerError('while scanning a block scalar', start_mark, 'expected indentation indicator in the range 1-9, but found 0', self.get_mark())
            self.forward()
    elif ch in '0123456789':
        increment = int(ch)
        if increment == 0:
            raise ScannerError('while scanning a block scalar', start_mark, 'expected indentation indicator in the range 1-9, but found 0', self.get_mark())
        self.forward()
        ch = self.peek()
        if ch in '+-':
            if ch == '+':
                chomping = True
            else:
                chomping = False
            self.forward()
    ch = self.peek()
    if ch not in '\x00 \r\n\x85\u2028\u2029':
        raise ScannerError('while scanning a block scalar', start_mark, 'expected chomping or indentation indicators, but found %r' % ch, self.get_mark())
    return (chomping, increment)