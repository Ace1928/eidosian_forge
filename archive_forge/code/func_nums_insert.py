from typing import Iterator, List, Optional, Tuple, cast
from ._protocols import PdfCommonDocProtocol
from ._utils import logger_warning
from .generic import ArrayObject, DictionaryObject, NullObject, NumberObject
def nums_insert(key: NumberObject, value: DictionaryObject, nums: ArrayObject) -> None:
    """
    Insert a key, value pair in a Nums array.

    See 7.9.7 "Number Trees".

    Args:
        key: number key of the entry
        value: value of the entry
        nums: Nums array to modify
    """
    if len(nums) % 2 != 0:
        raise ValueError('a nums like array must have an even number of elements')
    i = len(nums)
    while i != 0 and key <= nums[i - 2]:
        i = i - 2
    if i < len(nums) and key == nums[i]:
        nums[i + 1] = value
    else:
        nums.insert(i, key)
        nums.insert(i + 1, value)