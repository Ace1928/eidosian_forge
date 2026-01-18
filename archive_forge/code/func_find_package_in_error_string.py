import json
import os
import re
import subprocess
import sys
from typing import List, Optional, Set
def find_package_in_error_string(deps: List[str], line: str) -> Optional[str]:
    last_word = line.split(' ')[-1]
    if last_word in deps:
        return last_word
    for word in line.split(' '):
        if word.strip(',') in deps:
            return word
    return None