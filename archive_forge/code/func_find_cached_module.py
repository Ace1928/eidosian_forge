import sys
def find_cached_module(mod_name):
    return sys.modules.get(mod_name, None)