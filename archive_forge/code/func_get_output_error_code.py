from __future__ import annotations
import sys
from subprocess import PIPE, Popen
from typing import Any, Sequence
def get_output_error_code(cmd: str | Sequence[str]) -> tuple[str, str, Any]:
    """Get stdout, stderr, and exit code from running a command"""
    p = Popen(cmd, stdout=PIPE, stderr=PIPE)
    out, err = p.communicate()
    out_str = out.decode('utf8', 'replace')
    err_str = err.decode('utf8', 'replace')
    return (out_str, err_str, p.returncode)