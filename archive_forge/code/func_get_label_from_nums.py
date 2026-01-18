from typing import Iterator, List, Optional, Tuple, cast
from ._protocols import PdfCommonDocProtocol
from ._utils import logger_warning
from .generic import ArrayObject, DictionaryObject, NullObject, NumberObject
def get_label_from_nums(dictionary_object: DictionaryObject, index: int) -> str:
    nums = cast(ArrayObject, dictionary_object['/Nums'])
    i = 0
    value = None
    start_index = 0
    while i < len(nums):
        start_index = nums[i]
        value = nums[i + 1].get_object()
        if i + 2 == len(nums):
            break
        if nums[i + 2] > index:
            break
        i += 2
    m = {None: lambda n: '', '/D': lambda n: str(n), '/R': number2uppercase_roman_numeral, '/r': number2lowercase_roman_numeral, '/A': number2uppercase_letter, '/a': number2lowercase_letter}
    if not isinstance(value, dict):
        return str(index + 1)
    start = value.get('/St', 1)
    prefix = value.get('/P', '')
    return prefix + m[value.get('/S')](index - start_index + start)