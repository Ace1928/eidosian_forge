import os
import sys
import glob
import tempfile
import textwrap
import subprocess
import optparse
import logging
def make_command_file(path_to_debug_info, prefix_code='', no_import=False, skip_interpreter=False):
    if not no_import:
        pattern = os.path.join(path_to_debug_info, 'cython_debug', 'cython_debug_info_*')
        debug_files = glob.glob(pattern)
        if not debug_files:
            sys.exit('%s.\nNo debug files were found in %s. Aborting.' % (usage, os.path.abspath(path_to_debug_info)))
    fd, tempfilename = tempfile.mkstemp()
    f = os.fdopen(fd, 'w')
    try:
        f.write(prefix_code)
        f.write(textwrap.dedent('            # This is a gdb command file\n            # See https://sourceware.org/gdb/onlinedocs/gdb/Command-Files.html\n\n            set breakpoint pending on\n            set print pretty on\n\n            python\n            try:\n                # Activate virtualenv, if we were launched from one\n                import os\n                virtualenv = os.getenv(\'VIRTUAL_ENV\')\n                if virtualenv:\n                    path_to_activate_this_py = os.path.join(virtualenv, \'bin\', \'activate_this.py\')\n                    print("gdb command file: Activating virtualenv: %s; path_to_activate_this_py: %s" % (\n                        virtualenv, path_to_activate_this_py))\n                    with open(path_to_activate_this_py) as f:\n                        exec(f.read(), dict(__file__=path_to_activate_this_py))\n                from Cython.Debugger import libcython, libpython\n            except Exception as ex:\n                from traceback import print_exc\n                print("There was an error in Python code originating from the file ' + str(__file__) + '")\n                print("It used the Python interpreter " + str(sys.executable))\n                print_exc()\n                exit(1)\n            end\n            '))
        if no_import:
            pass
        else:
            if not skip_interpreter:
                path = os.path.join(path_to_debug_info, 'cython_debug', 'interpreter')
                interpreter_file = open(path)
                try:
                    interpreter = interpreter_file.read()
                finally:
                    interpreter_file.close()
                f.write('file %s\n' % interpreter)
            f.write('\n'.join(('cy import %s\n' % fn for fn in debug_files)))
            if not skip_interpreter:
                f.write(textwrap.dedent('                    python\n                    import sys\n                    try:\n                        gdb.lookup_type(\'PyModuleObject\')\n                    except RuntimeError:\n                        sys.stderr.write(\n                            "' + interpreter + ' was not compiled with debug symbols (or it was "\n                            "stripped). Some functionality may not work (properly).\\n")\n                    end\n                '))
            f.write('source .cygdbinit')
    finally:
        f.close()
    return tempfilename