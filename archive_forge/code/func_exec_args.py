import os
import re
import shutil
import sys
def exec_args(self, userargs):
    args = userargs[len(self.args):]
    if args:
        args[0] = os.path.basename(args[0])
    return args