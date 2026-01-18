from builtins import zip
from builtins import str
import os
import os.path as op
import sys
from xml.etree import cElementTree as ET
import pyxnat
def is_empty_resource(_res):
    """Check if resource contains any files"""
    f_count = 0
    for f_in in _res.files().fetchall('obj'):
        f_count += 1
        break
    return f_count == 0