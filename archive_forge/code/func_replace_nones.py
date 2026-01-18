import datetime
import locale
import re
import subprocess
import sys
import os
from collections import namedtuple
def replace_nones(dct, replacement='Could not collect'):
    for key in dct.keys():
        if dct[key] is not None:
            continue
        dct[key] = replacement
    return dct