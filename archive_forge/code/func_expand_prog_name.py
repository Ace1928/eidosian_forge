import sys, os
import textwrap
def expand_prog_name(self, s):
    return s.replace('%prog', self.get_prog_name())