import functools
import os
import subprocess
import sys
from mlflow.utils.os import is_windows
@classmethod
def from_completed_process(cls, process):
    lines = [f'Non-zero exit code: {process.returncode}', f'Command: {process.args}']
    if process.stdout:
        lines += ['', 'STDOUT:', process.stdout]
    if process.stderr:
        lines += ['', 'STDERR:', process.stderr]
    return cls('\n'.join(lines))