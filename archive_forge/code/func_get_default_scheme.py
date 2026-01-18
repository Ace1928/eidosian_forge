import os
import sys
from os.path import pardir, realpath
def get_default_scheme():
    return get_preferred_scheme('prefix')