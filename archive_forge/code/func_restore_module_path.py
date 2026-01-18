import sys
def restore_module_path():
    global _orig_sys_path
    sys.path = _orig_sys_path
    _orig_sys_path = None