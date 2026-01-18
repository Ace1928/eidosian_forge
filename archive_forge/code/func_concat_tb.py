from __future__ import annotations
from types import TracebackType
from typing import Any, ClassVar, cast
def concat_tb(head: TracebackType | None, tail: TracebackType | None) -> TracebackType | None:
    head_tbs = []
    pointer = head
    while pointer is not None:
        head_tbs.append(pointer)
        pointer = pointer.tb_next
    current_head = tail
    for head_tb in reversed(head_tbs):
        current_head = copy_tb(head_tb, tb_next=current_head)
    return current_head