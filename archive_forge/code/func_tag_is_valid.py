from operator import itemgetter
from typing import Any, List, Tuple, Dict, Sequence, Iterable, Optional, Mapping, Union
import warnings
import sys
from langcodes.tag_parser import LanguageTagError, parse_tag, normalize_characters
from langcodes.language_distance import tuple_distance_cached
from langcodes.data_dicts import (
def tag_is_valid(tag: Union[str, Language]) -> bool:
    """
    Determines whether a string is a valid language tag. This is similar to
    Language.get(tag).is_valid(), but can return False in the case where
    the tag doesn't parse.

    >>> tag_is_valid('ja')
    True
    >>> tag_is_valid('jp')
    False
    >>> tag_is_valid('spa-Latn-MX')
    True
    >>> tag_is_valid('spa-MX-Latn')
    False
    >>> tag_is_valid('')
    False
    >>> tag_is_valid('C.UTF-8')
    False
    """
    try:
        langdata = Language.get(tag)
        return langdata.is_valid()
    except LanguageTagError:
        return False