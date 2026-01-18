from _collections import deque
from collections import defaultdict
from functools import total_ordering
from typing import Any, Set, Dict, Union, NewType, Mapping, Tuple, Iterable
from interegular.utils import soft_repr
def nice_char_group(chars: Iterable[Union[str, _AnythingElseCls]]):
    out = []
    current_range = []
    for c in sorted(chars):
        if c is not anything_else and current_range and (ord(current_range[-1]) + 1 == ord(c)):
            current_range.append(c)
            continue
        if len(current_range) >= 2:
            out.append(f'{soft_repr(current_range[0])}-{soft_repr(current_range[-1])}')
        else:
            out.extend(map(soft_repr, current_range))
        current_range = [c]
    if len(current_range) >= 2:
        out.append(f'{soft_repr(current_range[0])}-{soft_repr(current_range[-1])}')
    else:
        out.extend(map(soft_repr, current_range))
    return ','.join(out)