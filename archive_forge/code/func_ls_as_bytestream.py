from __future__ import annotations
import os
import subprocess
from pathlib import Path
import typing as T
def ls_as_bytestream() -> bytes:
    if os.path.exists('.git'):
        return subprocess.run(['git', 'ls-tree', '-r', '--name-only', 'HEAD'], stdout=subprocess.PIPE).stdout
    files = [str(p) for p in Path('.').glob('**/*') if not p.is_dir() and (not next((x for x in p.parts if x.startswith('.')), None))]
    return '\n'.join(files).encode()