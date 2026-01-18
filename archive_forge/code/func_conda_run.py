import collections
import os
import shutil
import subprocess
def conda_run(*args):
    """Convenience method."""
    stdout, stderr, retcode = conda.cli.python_api.run_command(*args)
    if retcode:
        raise OSError(f'conda error: {str(args)}  retcode: {retcode}\n{stderr}')
    return stdout