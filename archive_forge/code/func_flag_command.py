import sys, re, curl, exceptions
from the command line first, then standard input.
def flag_command(self, func, line):
    if line.strip() in ('on', 'enable', 'yes'):
        func(True)
    elif line.strip() in ('off', 'disable', 'no'):
        func(False)
    else:
        print_stderr('linksys: unknown switch value')
    return 0