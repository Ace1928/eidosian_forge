import re
import numpy as np
@property
def fortran_format(self):
    r = '('
    if self.repeat:
        r += '%d' % self.repeat
    r += 'E%d.%d' % (self.width, self.significand)
    if self.min:
        r += 'E%d' % self.min
    return r + ')'