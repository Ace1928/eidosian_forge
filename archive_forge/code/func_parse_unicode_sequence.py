from __future__ import (absolute_import, division, print_function)
def parse_unicode_sequence(self):
    if self.index + 6 > self.length:
        raise InvalidLogFmt('Not enough space for unicode escape')
    if self.line[self.index:self.index + 2] != '\\u':
        raise InvalidLogFmt('Invalid unicode escape start')
    v = 0
    for i in range(self.index + 2, self.index + 6):
        v <<= 4
        try:
            v += _HEX_DICT[self.line[self.index]]
        except KeyError:
            raise InvalidLogFmt('Invalid unicode escape digit {digit!r}'.format(digit=self.line[self.index]))
    self.index += 6
    return chr(v)