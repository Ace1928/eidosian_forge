import atexit
import inspect
import os
import pprint
import re
import subprocess
import textwrap
def rm_temp():
    try:
        shutil.rmtree(tmp)
    except OSError:
        pass