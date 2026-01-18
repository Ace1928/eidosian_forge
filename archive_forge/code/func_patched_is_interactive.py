import sys
from _pydev_bundle import pydev_log
def patched_is_interactive():
    return matplotlib.rcParams['interactive']