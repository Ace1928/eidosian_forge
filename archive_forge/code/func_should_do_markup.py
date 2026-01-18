import os
import shutil
import sys
from typing import final
from typing import Literal
from typing import Optional
from typing import Sequence
from typing import TextIO
from .wcwidth import wcswidth
def should_do_markup(file: TextIO) -> bool:
    if os.environ.get('PY_COLORS') == '1':
        return True
    if os.environ.get('PY_COLORS') == '0':
        return False
    if os.environ.get('NO_COLOR'):
        return False
    if os.environ.get('FORCE_COLOR'):
        return True
    return hasattr(file, 'isatty') and file.isatty() and (os.environ.get('TERM') != 'dumb')