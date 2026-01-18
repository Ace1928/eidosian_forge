import sys
import os
def find_in_pythonpath(module_name):
    found_at = []
    parts = module_name.split('.')
    for path in sys.path:
        target = os.path.join(path, *parts)
        target_py = target + '.py'
        if os.path.isdir(target):
            found_at.append(target)
        if os.path.exists(target_py):
            found_at.append(target_py)
    return found_at