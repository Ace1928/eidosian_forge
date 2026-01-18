import sys
from os import environ
def init_gl(allowed=[], ignored=[]):
    gl_init_symbols(allowed, ignored)
    print_gl_version()
    gl_init_resources()