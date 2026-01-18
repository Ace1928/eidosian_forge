import sys
from .colorama.win32 import windll
from .colorama.winterm import WinColor, WinStyle, WinTerm
def cout(*args):
    """Shorthand for cprint('stdout', ...)"""
    cprint('stdout', *args)