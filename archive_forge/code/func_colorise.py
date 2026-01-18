import argparse
import collections
import functools
import sys
import time
from oslo_serialization import jsonutils
from oslo_utils import importutils
from oslo_log import log
def colorise(key, text=None):
    if text is None:
        text = key
    if not _USE_COLOR:
        return text
    colors = {'exc': ('red', ['reverse', 'bold']), 'FATAL': ('red', ['reverse', 'bold']), 'ERROR': ('red', ['bold']), 'WARNING': ('yellow', ['bold']), 'WARN': ('yellow', ['bold']), 'INFO': ('white', ['bold'])}
    color, attrs = colors.get(key, ('', []))
    if color:
        return termcolor.colored(text, color=color, attrs=attrs)
    return text