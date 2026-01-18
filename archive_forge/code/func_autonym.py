from operator import itemgetter
from typing import Any, List, Tuple, Dict, Sequence, Iterable, Optional, Mapping, Union
import warnings
import sys
from langcodes.tag_parser import LanguageTagError, parse_tag, normalize_characters
from langcodes.language_distance import tuple_distance_cached
from langcodes.data_dicts import (
def autonym(self, max_distance: int=9) -> str:
    """
        Give the display name of this language *in* this language.
        Requires that `language_data` is installed.

        >>> Language.get('fr').autonym()
        'français'
        >>> Language.get('es').autonym()
        'español'
        >>> Language.get('ja').autonym()
        '日本語'

        This uses the `display_name()` method, so it can include the name of a
        script or territory when appropriate.

        >>> Language.get('en-AU').autonym()
        'English (Australia)'
        >>> Language.get('sr-Latn').autonym()
        'srpski (latinica)'
        >>> Language.get('sr-Cyrl').autonym()
        'српски (ћирилица)'
        >>> Language.get('pa').autonym()
        'ਪੰਜਾਬੀ'
        >>> Language.get('pa-Arab').autonym()
        'پنجابی (عربی)'

        This only works for language codes that CLDR has locale data for. You
        can't ask for the autonym of 'ja-Latn' and get 'nihongo (rōmaji)'.
        """
    lang = self.prefer_macrolanguage()
    return lang.display_name(language=lang, max_distance=max_distance)