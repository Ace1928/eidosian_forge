import sys, os
import textwrap
def get_prog_name(self):
    if self.prog is None:
        return os.path.basename(sys.argv[0])
    else:
        return self.prog