import functools
import logging
import subprocess
from typing import List, Optional, TypeVar
from typing_extensions import TypedDict
def exec_git(command: List[str]) -> Optional[str]:
    try:
        return subprocess.check_output(['git'] + command, encoding='utf-8', stderr=subprocess.DEVNULL).strip()
    except BaseException:
        return None