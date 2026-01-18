import os
import subprocess
from .errors import HookError
def clean_msg(success, *args):
    if success:
        with open(args[0], 'rb') as f:
            new_msg = f.read()
        os.unlink(args[0])
        return new_msg
    os.unlink(args[0])