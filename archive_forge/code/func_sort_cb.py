import atexit
import inspect
import os
import pprint
import re
import subprocess
import textwrap
def sort_cb(k):
    if isinstance(k, str):
        return self.feature_supported[k]['interest']
    rank = max([self.feature_supported[f]['interest'] for f in k])
    rank += len(k) - 1
    return rank