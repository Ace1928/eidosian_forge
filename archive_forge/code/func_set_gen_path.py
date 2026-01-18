import comhack
import sys
import __builtin__
def set_gen_path(path):
    """
    Set the gencache Path
    If not set, all Modules (win32com) will be generated to support/gen_py of your apllication.
    """
    import os
    import win32com
    import win32com.gen_py
    path = os.path.abspath(path)
    if not os.path.exists(path):
        os.makedirs(path)
    frozen = sys.__dict__.get('frozen', 0)
    win32com.gen_py.__path__ = [path]
    win32com.__gen_path__ = path
    if not frozen:
        return
    __builtin__.__import__ = _myimport