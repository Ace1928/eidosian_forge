import os
import platform
import subprocess
import errno
import time
import sys
import unittest
import tempfile
def get_conn_maxsize(self, which, maxsize):
    if maxsize is None:
        maxsize = 1024
    elif maxsize < 1:
        maxsize = 1
    return (getattr(self, which), maxsize)