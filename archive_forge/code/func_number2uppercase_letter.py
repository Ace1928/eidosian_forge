from typing import Iterator, List, Optional, Tuple, cast
from ._protocols import PdfCommonDocProtocol
from ._utils import logger_warning
from .generic import ArrayObject, DictionaryObject, NullObject, NumberObject
def number2uppercase_letter(number: int) -> str:
    if number <= 0:
        raise ValueError('Expecting a positive number')
    alphabet = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
    rep = ''
    while number > 0:
        remainder = number % 26
        if remainder == 0:
            remainder = 26
        rep = alphabet[remainder - 1] + rep
        number -= remainder
        number = number // 26
    return rep