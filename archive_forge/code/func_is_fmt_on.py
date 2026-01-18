import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Collection, Final, Iterator, List, Optional, Tuple, Union
from black.mode import Mode, Preview
from black.nodes import (
from blib2to3.pgen2 import token
from blib2to3.pytree import Leaf, Node
def is_fmt_on(container: LN) -> bool:
    """Determine whether formatting is switched on within a container.
    Determined by whether the last `# fmt:` comment is `on` or `off`.
    """
    fmt_on = False
    for comment in list_comments(container.prefix, is_endmarker=False):
        if comment.value in FMT_ON:
            fmt_on = True
        elif comment.value in FMT_OFF:
            fmt_on = False
    return fmt_on