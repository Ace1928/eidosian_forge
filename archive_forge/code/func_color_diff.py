import json
import re
import tempfile
from typing import Any, List, Optional
from click import echo, style
from mypy_extensions import mypyc_attr
def color_diff(contents: str) -> str:
    """Inject the ANSI color codes to the diff."""
    lines = contents.split('\n')
    for i, line in enumerate(lines):
        if line.startswith('+++') or line.startswith('---'):
            line = '\x1b[1m' + line + '\x1b[0m'
        elif line.startswith('@@'):
            line = '\x1b[36m' + line + '\x1b[0m'
        elif line.startswith('+'):
            line = '\x1b[32m' + line + '\x1b[0m'
        elif line.startswith('-'):
            line = '\x1b[31m' + line + '\x1b[0m'
        lines[i] = line
    return '\n'.join(lines)