import re
import sys
import subprocess
def parse_cmd():
    """Parses the command-line arguments.

libfile, deffile = parse_cmd()"""
    if len(sys.argv) == 3:
        if sys.argv[1][-4:] == '.lib' and sys.argv[2][-4:] == '.def':
            libfile, deffile = sys.argv[1:]
        elif sys.argv[1][-4:] == '.def' and sys.argv[2][-4:] == '.lib':
            deffile, libfile = sys.argv[1:]
        else:
            print("I'm assuming that your first argument is the library")
            print('and the second is the DEF file.')
    elif len(sys.argv) == 2:
        if sys.argv[1][-4:] == '.def':
            deffile = sys.argv[1]
            libfile = 'python%s.lib' % py_ver
        elif sys.argv[1][-4:] == '.lib':
            deffile = None
            libfile = sys.argv[1]
    else:
        libfile = 'python%s.lib' % py_ver
        deffile = None
    return (libfile, deffile)