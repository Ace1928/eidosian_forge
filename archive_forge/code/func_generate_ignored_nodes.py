import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Collection, Final, Iterator, List, Optional, Tuple, Union
from black.mode import Mode, Preview
from black.nodes import (
from blib2to3.pgen2 import token
from blib2to3.pytree import Leaf, Node
def generate_ignored_nodes(leaf: Leaf, comment: ProtoComment, mode: Mode) -> Iterator[LN]:
    """Starting from the container of `leaf`, generate all leaves until `# fmt: on`.

    If comment is skip, returns leaf only.
    Stops at the end of the block.
    """
    if _contains_fmt_skip_comment(comment.value, mode):
        yield from _generate_ignored_nodes_from_fmt_skip(leaf, comment)
        return
    container: Optional[LN] = container_of(leaf)
    while container is not None and container.type != token.ENDMARKER:
        if is_fmt_on(container):
            return
        if children_contains_fmt_on(container):
            for index, child in enumerate(container.children):
                if isinstance(child, Leaf) and is_fmt_on(child):
                    if child.type in CLOSING_BRACKETS:
                        yield child
                    return
                if child.type == token.INDENT and index < len(container.children) - 1 and children_contains_fmt_on(container.children[index + 1]):
                    return
                if children_contains_fmt_on(child):
                    return
                yield child
        else:
            if container.type == token.DEDENT and container.next_sibling is None:
                return
            yield container
            container = container.next_sibling