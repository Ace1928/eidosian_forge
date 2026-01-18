from operator import itemgetter
from typing import Any, List, Tuple, Dict, Sequence, Iterable, Optional, Mapping, Union
import warnings
import sys
from langcodes.tag_parser import LanguageTagError, parse_tag, normalize_characters
from langcodes.language_distance import tuple_distance_cached
from langcodes.data_dicts import (
def territory_name(self, language: Union[str, 'Language']=DEFAULT_LANGUAGE, max_distance: int=25) -> str:
    """
        Describe the territory part of the language tag in a natural language.
        Requires that `language_data` is installed.
        """
    return self._get_name('territory', language, max_distance)