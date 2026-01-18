import sys
import traceback
from pprint import pformat
from pathlib import Path
from IPython.core import ultratb
from IPython.core.release import author_email
from IPython.utils.sysinfo import sys_info
from IPython.utils.py3compat import input
from IPython.core.release import __version__ as version
from typing import Optional
def crash_handler_lite(etype, evalue, tb):
    """a light excepthook, adding a small message to the usual traceback"""
    traceback.print_exception(etype, evalue, tb)
    from IPython.core.interactiveshell import InteractiveShell
    if InteractiveShell.initialized():
        config = '%config '
    else:
        config = 'c.'
    print(_lite_message_template.format(email=author_email, config=config, version=version), file=sys.stderr)