import datetime
import locale
import re
import subprocess
import sys
import os
from collections import namedtuple
def maybe_start_on_next_line(string):
    if string is not None and len(string.split('\n')) > 1:
        return '\n{}\n'.format(string)
    return string