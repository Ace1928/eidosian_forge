import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Collection, Final, Iterator, List, Optional, Tuple, Union
from black.mode import Mode, Preview
from black.nodes import (
from blib2to3.pgen2 import token
from blib2to3.pytree import Leaf, Node
@lru_cache(maxsize=4096)
def list_comments(prefix: str, *, is_endmarker: bool) -> List[ProtoComment]:
    """Return a list of :class:`ProtoComment` objects parsed from the given `prefix`."""
    result: List[ProtoComment] = []
    if not prefix or '#' not in prefix:
        return result
    consumed = 0
    nlines = 0
    ignored_lines = 0
    form_feed = False
    for index, full_line in enumerate(re.split('\r?\n', prefix)):
        consumed += len(full_line) + 1
        match = re.match('^(\\s*)(\\S.*|)$', full_line)
        assert match
        whitespace, line = match.groups()
        if not line:
            nlines += 1
            if '\x0c' in full_line:
                form_feed = True
        if not line.startswith('#'):
            if line.endswith('\\'):
                ignored_lines += 1
            continue
        if index == ignored_lines and (not is_endmarker):
            comment_type = token.COMMENT
        else:
            comment_type = STANDALONE_COMMENT
        comment = make_comment(line)
        result.append(ProtoComment(type=comment_type, value=comment, newlines=nlines, consumed=consumed, form_feed=form_feed, leading_whitespace=whitespace))
        form_feed = False
        nlines = 0
    return result