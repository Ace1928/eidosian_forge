import argparse
import glob
import os
import re
import subprocess
from textwrap import dedent
from typing import Iterable, Optional
def process_package_name(lines: Iterable[str], package_name: str) -> Iterable[str]:
    need_rename = package_name != DEFAULT_PACKAGE_NAME
    for line in lines:
        m = IMPORT_REGEX.match(line) if need_rename else None
        if m:
            include_name = m.group(2)
            ml = ML_REGEX.match(include_name)
            if ml:
                include_name = f'{ml.group(1)}_{package_name}-ml'
            else:
                include_name = f'{include_name}_{package_name}'
            yield (m.group(1) + f'import "{include_name}.proto";')
        else:
            yield PACKAGE_NAME_REGEX.sub(package_name, line)