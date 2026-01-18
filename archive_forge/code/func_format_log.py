import ast
import collections
import os
import re
import shutil
import sys
import tempfile
import traceback
import pasta
def format_log(self, log, in_filename):
    log_string = '%d:%d: %s: %s' % (log[1], log[2], log[0], log[3])
    if in_filename:
        return in_filename + ':' + log_string
    else:
        return log_string