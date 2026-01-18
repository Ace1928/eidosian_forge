import os
import sys
def has_ansi_colors():
    if sys.platform == 'win32':
        return False
    if not sys.stdout.isatty():
        return False
    import curses
    try:
        curses.setupterm()
    except curses.error:
        return False
    return bool(curses.tigetstr('setaf'))