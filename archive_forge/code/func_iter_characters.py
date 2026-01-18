from sys import maxunicode
from typing import cast, Iterable, Iterator, List, MutableSet, Union, Optional
from .unicode_categories import RAW_UNICODE_CATEGORIES
from .codepoints import CodePoint, code_point_order, code_point_repr, \
def iter_characters(self) -> Iterator[str]:
    return map(chr, self.__iter__())