import json
import os
import re
import subprocess
import sys
from typing import List, Optional, Set
def add_version_to_package_name(deps: List[str], package: str) -> Optional[str]:
    """Add the associated version to a package name.

    For example: `my-package` -> `my-package==1.0.0`
    """
    for dep in deps:
        if dep.split('==')[0] == package:
            return dep
    return None