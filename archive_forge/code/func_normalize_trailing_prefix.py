import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Collection, Final, Iterator, List, Optional, Tuple, Union
from black.mode import Mode, Preview
from black.nodes import (
from blib2to3.pgen2 import token
from blib2to3.pytree import Leaf, Node
def normalize_trailing_prefix(leaf: LN, total_consumed: int) -> None:
    """Normalize the prefix that's left over after generating comments.

    Note: don't use backslashes for formatting or you'll lose your voting rights.
    """
    remainder = leaf.prefix[total_consumed:]
    if '\\' not in remainder:
        nl_count = remainder.count('\n')
        form_feed = '\x0c' in remainder and remainder.endswith('\n')
        leaf.prefix = make_simple_prefix(nl_count, form_feed)
        return
    leaf.prefix = ''