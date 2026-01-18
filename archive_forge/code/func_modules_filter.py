import os
import importlib
import sys
def modules_filter(module):
    return all((not module.startswith('_'), not module.startswith('python-'), os.path.isdir(os.path.join(_extra_submodules_init_path, module))))