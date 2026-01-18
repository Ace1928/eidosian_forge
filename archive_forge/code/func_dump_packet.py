from .charset import MBLENGTH
from .constants import FIELD_TYPE, SERVER_STATUS
from . import err
import struct
import sys
def dump_packet(data):

    def printable(data):
        if 32 <= data < 127:
            return chr(data)
        return '.'
    try:
        print('packet length:', len(data))
        for i in range(1, 7):
            f = sys._getframe(i)
            print('call[%d]: %s (line %d)' % (i, f.f_code.co_name, f.f_lineno))
        print('-' * 66)
    except ValueError:
        pass
    dump_data = [data[i:i + 16] for i in range(0, min(len(data), 256), 16)]
    for d in dump_data:
        print(' '.join((f'{x:02X}' for x in d)) + '   ' * (16 - len(d)) + ' ' * 2 + ''.join((printable(x) for x in d)))
    print('-' * 66)
    print()