from __future__ import annotations
import os
import subprocess
def git_revision(dir: str) -> str:
    """Get the SHA-1 of the HEAD of a git repository."""
    return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], cwd=dir).decode('utf-8').strip()