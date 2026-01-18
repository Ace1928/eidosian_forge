from typing import Iterator, List, Optional, Tuple, cast
from ._protocols import PdfCommonDocProtocol
from ._utils import logger_warning
from .generic import ArrayObject, DictionaryObject, NullObject, NumberObject
def number2uppercase_roman_numeral(num: int) -> str:
    roman = [(1000, 'M'), (900, 'CM'), (500, 'D'), (400, 'CD'), (100, 'C'), (90, 'XC'), (50, 'L'), (40, 'XL'), (10, 'X'), (9, 'IX'), (5, 'V'), (4, 'IV'), (1, 'I')]

    def roman_num(num: int) -> Iterator[str]:
        for decimal, roman_repr in roman:
            x, _ = divmod(num, decimal)
            yield (roman_repr * x)
            num -= decimal * x
            if num <= 0:
                break
    return ''.join(list(roman_num(num)))