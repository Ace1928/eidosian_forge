import sys
import doctest
import re
import types
from .numeric_output_checker import NumericOutputChecker
def cyopengl_works():
    if not _gui_status['cyopengl']:
        return False
    if _gui_status['tk'] and (not Tk_._default_root):
        try:
            root = Tk_.Tk()
            if sys.platform not in ('linux', 'linux2'):
                root.withdraw()
        except:
            _gui_status['tk'] = _gui_status['cyopengl'] = False
    return _gui_status['cyopengl']