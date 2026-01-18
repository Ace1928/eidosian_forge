import time
import os
def get_key_name(self, fullpath, prefix):
    key_name = fullpath[len(prefix):]
    l = key_name.split(os.sep)
    return '/'.join(l)