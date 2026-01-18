import datetime
import errno
import html
import json
import os
import random
import shlex
import textwrap
import time
from tensorboard import manager
def format_stream(name, value):
    if value == '':
        return ''
    elif value is None:
        return '\n<could not read %s>' % name
    else:
        return '\nContents of %s:\n%s' % (name, value.strip())