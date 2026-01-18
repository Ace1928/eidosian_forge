from __future__ import print_function
import gdb
import os
import locale
import sys
import sys
import libpython
import re
import warnings
import tempfile
import functools
import textwrap
import itertools
import traceback
def register_defines():
    source_gdb_script(textwrap.dedent('        define py-step\n        -py-step\n        end\n\n        define py-next\n        -py-next\n        end\n\n        document py-step\n        %s\n        end\n\n        document py-next\n        %s\n        end\n    ') % (PyStep.__doc__, PyNext.__doc__))