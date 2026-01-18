import os
import re
import shutil
import sys
from typing import Dict, Pattern
def term_width_line(text: str) -> str:
    if not codes:
        return text + '\n'
    else:
        return text.ljust(_tw + len(text) - len(_ansi_re.sub('', text))) + '\r'