from __future__ import (absolute_import, division, print_function)
@contextlib.contextmanager
def monitor_sys_modules(path, messages):
    """Monitor sys.modules for unwanted changes, reverting any additions made to our own namespaces."""
    snapshot = sys.modules.copy()
    try:
        yield
    finally:
        check_sys_modules(path, snapshot, messages)
        for key in set(sys.modules.keys()) - set(snapshot.keys()):
            if is_name_in_namepace(key, ('ansible', 'ansible_collections')):
                del sys.modules[key]