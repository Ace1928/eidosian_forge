import re
import html
import contextlib
from typing import Optional
def cleanup_dots(text: str) -> str:
    """
    Cleans up the dots found in table of contents

    INTRODUCTION............................................ > INTRODUCTION.
    """
    return re.sub('\\.+', '.', text).strip()