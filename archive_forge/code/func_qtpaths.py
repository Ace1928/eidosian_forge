import sys
import subprocess
from pathlib import Path
from typing import List, Dict, Optional
from . import opt_dry_run, opt_quiet, QTPATHS_CMD, PROJECT_FILE_SUFFIX
def qtpaths() -> Dict[str, str]:
    """Run qtpaths and return a dict of values."""
    global _qtpaths_info
    if not _qtpaths_info:
        output = subprocess.check_output([QTPATHS_CMD, '--query'])
        for line in output.decode('utf-8').split('\n'):
            tokens = line.strip().split(':', maxsplit=1)
            if len(tokens) == 2:
                _qtpaths_info[tokens[0]] = tokens[1]
    return _qtpaths_info