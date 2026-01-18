import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Collection, Final, Iterator, List, Optional, Tuple, Union
from black.mode import Mode, Preview
from black.nodes import (
from blib2to3.pgen2 import token
from blib2to3.pytree import Leaf, Node
def make_comment(content: str) -> str:
    """Return a consistently formatted comment from the given `content` string.

    All comments (except for "##", "#!", "#:", '#'") should have a single
    space between the hash sign and the content.

    If `content` didn't start with a hash sign, one is provided.
    """
    content = content.rstrip()
    if not content:
        return '#'
    if content[0] == '#':
        content = content[1:]
    NON_BREAKING_SPACE = '\xa0'
    if content and content[0] == NON_BREAKING_SPACE and (not content.lstrip().startswith('type:')):
        content = ' ' + content[1:]
    if content and content[0] not in COMMENT_EXCEPTIONS:
        content = ' ' + content
    return '#' + content