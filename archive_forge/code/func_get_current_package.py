import json
import os
import re
import subprocess
import sys
from typing import List, Optional, Set
def get_current_package(line: str, deps: List[str], current_pkg: Optional[str]) -> Optional[str]:
    """Tries to pull a package name from the line.

    Used to keep track of what the currently-installing package is,
    in case an error message isn't on the same line as the package
    """
    if line.startswith('Collecting'):
        return line.split(' ')[1]
    elif line.strip().startswith('Building wheel') and line.strip().endswith("finished with status 'error'"):
        return add_version_to_package_name(deps, line.strip().split(' ')[3])
    elif line.strip().startswith('Running setup.py install') and line.strip().endswith("finished with status 'error'"):
        return add_version_to_package_name(deps, line.strip().split(' ')[4][:-1])
    return current_pkg