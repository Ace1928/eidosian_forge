import sys
import os
import os.path
import tempfile
import zipfile
from xml.dom import minidom
import time
import re
import copy
import itertools
import docutils
from docutils import frontend, nodes, utils, writers, languages
from docutils.readers import standalone
from docutils.transforms import references
def convert_to_cm(self, size):
    """Convert various units to centimeters.

        Note that a call to this method should be wrapped in:
            try: except ValueError:
        """
    size = size.strip()
    if size.endswith('px'):
        size = float(size[:-2]) * 0.026
    elif size.endswith('in'):
        size = float(size[:-2]) * 2.54
    elif size.endswith('pt'):
        size = float(size[:-2]) * 0.035
    elif size.endswith('pc'):
        size = float(size[:-2]) * 2.371
    elif size.endswith('mm'):
        size = float(size[:-2]) * 0.1
    elif size.endswith('cm'):
        size = float(size[:-2])
    else:
        raise ValueError('unknown unit type')
    unit = 'cm'
    return (size, unit)