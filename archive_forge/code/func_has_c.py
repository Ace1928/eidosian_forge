import os
from .compilation import compile_run_strings
from .util import CompilerNotFoundError
def has_c():
    if not hasattr(has_c, 'result'):
        try:
            (stdout, stderr), info = compile_run_strings([('main.c', '#include <stdio.h>\nint main(){\nprintf("hello world\\n");\nreturn 0;\n}')], clean=True)
        except CompilerNotFoundError:
            has_c.result = False
            if os.environ.get('SYMPY_STRICT_COMPILER_CHECKS', '0') == '1':
                raise
        else:
            if info['exit_status'] != os.EX_OK or 'hello world' not in stdout:
                if os.environ.get('SYMPY_STRICT_COMPILER_CHECKS', '0') == '1':
                    raise ValueError('Failed to compile test program:\n%s\n%s\n' % (stdout, stderr))
                has_c.result = False
            else:
                has_c.result = True
    return has_c.result