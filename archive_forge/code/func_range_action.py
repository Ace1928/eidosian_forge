import argparse
import os
from time import time
from pyzstd import compress_stream, decompress_stream, \
def range_action(min, max, bits_msg=False):

    class RangeAction(argparse.Action):

        def __call__(self, parser, args, values, option_string=None):
            try:
                v = int(values)
            except:
                raise TypeError('{} should be an integer'.format(option_string))
            if not min <= v <= max:
                if bits_msg:
                    bits = 'in {}-bit build, '.format(PYZSTD_CONFIG[0])
                else:
                    bits = ''
                msg = '{}{} value should: {} <= v <= {}. provided value is {}.'.format(bits, option_string, min, max, v)
                raise ValueError(msg)
            setattr(args, self.dest, v)
    return RangeAction