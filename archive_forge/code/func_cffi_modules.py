import os
import sys
def cffi_modules(dist, attr, value):
    assert attr == 'cffi_modules'
    if isinstance(value, basestring):
        value = [value]
    for cffi_module in value:
        add_cffi_module(dist, cffi_module)