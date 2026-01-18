import sys
def sysexit(stat, mode):
    raise SystemExit(stat, f'Mode = {mode}')