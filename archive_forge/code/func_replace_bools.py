import datetime
import locale
import re
import subprocess
import sys
import os
from collections import namedtuple
def replace_bools(dct, true='Yes', false='No'):
    for key in dct.keys():
        if dct[key] is True:
            dct[key] = true
        elif dct[key] is False:
            dct[key] = false
    return dct