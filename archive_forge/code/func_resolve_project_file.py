import sys
import subprocess
from pathlib import Path
from typing import List, Dict, Optional
from . import opt_dry_run, opt_quiet, QTPATHS_CMD, PROJECT_FILE_SUFFIX
def resolve_project_file(cmdline: str) -> Optional[Path]:
    """Return the project file from the command  line value, either
    from the file argument or directory"""
    project_file = Path(cmdline).resolve() if cmdline else Path.cwd()
    if project_file.is_file():
        return project_file
    if project_file.is_dir():
        for m in project_file.glob(f'*{PROJECT_FILE_SUFFIX}'):
            return m
    return None