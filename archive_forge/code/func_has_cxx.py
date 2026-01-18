import os
from .compilation import compile_run_strings
from .util import CompilerNotFoundError
def has_cxx():
    if not hasattr(has_cxx, 'result'):
        try:
            (stdout, stderr), info = compile_run_strings([('main.cxx', '#include <iostream>\nint main(){\nstd::cout << "hello world" << std::endl;\n}')], clean=True)
        except CompilerNotFoundError:
            has_cxx.result = False
            if os.environ.get('SYMPY_STRICT_COMPILER_CHECKS', '0') == '1':
                raise
        else:
            if info['exit_status'] != os.EX_OK or 'hello world' not in stdout:
                if os.environ.get('SYMPY_STRICT_COMPILER_CHECKS', '0') == '1':
                    raise ValueError('Failed to compile test program:\n%s\n%s\n' % (stdout, stderr))
                has_cxx.result = False
            else:
                has_cxx.result = True
    return has_cxx.result