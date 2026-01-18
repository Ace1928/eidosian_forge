from operator import itemgetter
from typing import Any, List, Tuple, Dict, Sequence, Iterable, Optional, Mapping, Union
import warnings
import sys
from langcodes.tag_parser import LanguageTagError, parse_tag, normalize_characters
from langcodes.language_distance import tuple_distance_cached
from langcodes.data_dicts import (
def prefer_macrolanguage(self) -> 'Language':
    """
        BCP 47 doesn't specify what to do with macrolanguages and the languages
        they contain. The Unicode CLDR, on the other hand, says that when a
        macrolanguage has a dominant standardized language, the macrolanguage
        code should be used for that language. For example, Mandarin Chinese
        is 'zh', not 'cmn', according to Unicode, and Malay is 'ms', not 'zsm'.

        This isn't a rule you'd want to follow in all cases -- for example, you may
        want to be able to specifically say that 'ms' (the Malay macrolanguage)
        contains both 'zsm' (Standard Malay) and 'id' (Indonesian). But applying
        this rule helps when interoperating with the Unicode CLDR.

        So, applying `prefer_macrolanguage` to a Language object will
        return a new object, replacing the language with the macrolanguage if
        it is the dominant language within that macrolanguage. It will leave
        non-dominant languages that have macrolanguages alone.

        >>> Language.get('arb').prefer_macrolanguage()
        Language.make(language='ar')

        >>> Language.get('cmn-Hant').prefer_macrolanguage()
        Language.make(language='zh', script='Hant')

        >>> Language.get('yue-Hant').prefer_macrolanguage()
        Language.make(language='yue', script='Hant')
        """
    if self._macrolanguage is not None:
        return self._macrolanguage
    language = self.language or 'und'
    if language in NORMALIZED_MACROLANGUAGES:
        self._macrolanguage = self.update_dict({'language': NORMALIZED_MACROLANGUAGES[language]})
    else:
        self._macrolanguage = self
    return self._macrolanguage