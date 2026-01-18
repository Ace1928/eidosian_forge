from typing import List, NamedTuple, Optional
def parts_with(val: int) -> List[str]:
    parts = []
    if top == val:
        parts.append('top')
    if bottom == val:
        parts.append('bottom')
    if left == val:
        parts.append('left')
    if right == val:
        parts.append('right')
    return parts