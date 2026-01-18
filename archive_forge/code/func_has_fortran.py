import os
from .compilation import compile_run_strings
from .util import CompilerNotFoundError
def has_fortran():
    if not hasattr(has_fortran, 'result'):
        try:
            (stdout, stderr), info = compile_run_strings([('main.f90', 'program foo\nprint *, "hello world"\nend program')], clean=True)
        except CompilerNotFoundError:
            has_fortran.result = False
            if os.environ.get('SYMPY_STRICT_COMPILER_CHECKS', '0') == '1':
                raise
        else:
            if info['exit_status'] != os.EX_OK or 'hello world' not in stdout:
                if os.environ.get('SYMPY_STRICT_COMPILER_CHECKS', '0') == '1':
                    raise ValueError('Failed to compile test program:\n%s\n%s\n' % (stdout, stderr))
                has_fortran.result = False
            else:
                has_fortran.result = True
    return has_fortran.result