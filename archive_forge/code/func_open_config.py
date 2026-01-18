import contextlib
import os
import shutil
import stat
import sys
@contextlib.contextmanager
def open_config(filename):
    if sys.version_info >= (3, 2):
        cfg = configparser.ConfigParser()
    else:
        cfg = configparser.SafeConfigParser()
    cfg.read(filename)
    yield cfg
    with open(filename, 'w') as fp:
        cfg.write(fp)