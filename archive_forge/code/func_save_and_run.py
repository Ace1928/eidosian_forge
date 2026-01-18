import sys
import textwrap
from pathlib import Path
def save_and_run(program, base_dir=None, output=None, file_name=None, optimized=False):
    """
    safe and run a python program, thereby circumventing any restrictions on module level
    imports
    """
    from subprocess import check_output, STDOUT, CalledProcessError
    if not hasattr(base_dir, 'hash'):
        base_dir = Path(str(base_dir))
    if file_name is None:
        file_name = 'safe_and_run_tmp.py'
    file_name = base_dir / file_name
    file_name.write_text(dedent(program))
    try:
        cmd = [sys.executable]
        if optimized:
            cmd.append('-O')
        cmd.append(str(file_name))
        print('running:', *cmd)
        res = check_output(cmd, stderr=STDOUT, universal_newlines=True)
        if output is not None:
            if '__pypy__' in sys.builtin_module_names:
                res = res.splitlines(True)
                res = [line for line in res if 'no version info' not in line]
                res = ''.join(res)
            print('result:  ', res, end='')
            print('expected:', output, end='')
            assert res == output
    except CalledProcessError as exception:
        print("##### Running '{} {}' FAILED #####".format(sys.executable, file_name))
        print(exception.output)
        return exception.returncode
    return 0