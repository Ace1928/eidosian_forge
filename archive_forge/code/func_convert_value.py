import sys, os
import textwrap
def convert_value(self, opt, value):
    if value is not None:
        if self.nargs == 1:
            return self.check_value(opt, value)
        else:
            return tuple([self.check_value(opt, v) for v in value])