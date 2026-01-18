import rpy2.rlike.indexing as rli
from typing import Any
from typing import Iterable
from typing import List
from typing import Optional
from typing import Tuple
@staticmethod
def from_items(tagval):
    res = TaggedList([])
    for k, v in tagval.items():
        res.append(v, tag=k)
    return res