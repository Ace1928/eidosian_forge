from __future__ import print_function
def get_function_module(name):
    for mod, cb in module_callbacks().items():
        if cb(name):
            return mod
    if '.' in name:
        return name.split('.')[0]
    else:
        return 'basic'