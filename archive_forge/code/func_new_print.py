import builtins
import copy
import os
import pickle
import contextlib
import subprocess
import socket
import parlai.utils.logging as logging
def new_print(*args, **kwargs):
    if suppress:
        return
    elif prefix:
        return builtin_print(prefix, *args, **kwargs)
    else:
        return builtin_print(*args, **kwargs)