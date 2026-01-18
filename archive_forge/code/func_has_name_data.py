from operator import itemgetter
from typing import Any, List, Tuple, Dict, Sequence, Iterable, Optional, Mapping, Union
import warnings
import sys
from langcodes.tag_parser import LanguageTagError, parse_tag, normalize_characters
from langcodes.language_distance import tuple_distance_cached
from langcodes.data_dicts import (
def has_name_data(self) -> bool:
    """
        Return True when we can name languages in this language. Requires
        `language_data` to be installed.

        This is true when the language, or one of its 'broader' versions, is in
        the list of CLDR target languages.

        >>> Language.get('fr').has_name_data()
        True
        >>> Language.get('so').has_name_data()
        True
        >>> Language.get('enc').has_name_data()
        False
        >>> Language.get('und').has_name_data()
        False
        """
    try:
        from language_data.name_data import LANGUAGES_WITH_NAME_DATA
    except ImportError:
        print(LANGUAGE_NAME_IMPORT_MESSAGE, file=sys.stdout)
        raise
    matches = set(self.broader_tags()) & LANGUAGES_WITH_NAME_DATA
    return bool(matches)