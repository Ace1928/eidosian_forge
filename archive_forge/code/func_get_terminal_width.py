import os
import shutil
import sys
from typing import final
from typing import Literal
from typing import Optional
from typing import Sequence
from typing import TextIO
from .wcwidth import wcswidth
def get_terminal_width() -> int:
    width, _ = shutil.get_terminal_size(fallback=(80, 24))
    if width < 40:
        width = 80
    return width