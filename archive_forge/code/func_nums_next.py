from typing import Iterator, List, Optional, Tuple, cast
from ._protocols import PdfCommonDocProtocol
from ._utils import logger_warning
from .generic import ArrayObject, DictionaryObject, NullObject, NumberObject
def nums_next(key: NumberObject, nums: ArrayObject) -> Tuple[Optional[NumberObject], Optional[DictionaryObject]]:
    """
    Return the (key, value) pair of the entry after the given one.

    See 7.9.7 "Number Trees".

    Args:
        key: number key of the entry
        nums: Nums array
    """
    if len(nums) % 2 != 0:
        raise ValueError('a nums like array must have an even number of elements')
    i = nums.index(key) + 2
    if i < len(nums):
        return (nums[i], nums[i + 1])
    else:
        return (None, None)