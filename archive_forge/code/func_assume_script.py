from operator import itemgetter
from typing import Any, List, Tuple, Dict, Sequence, Iterable, Optional, Mapping, Union
import warnings
import sys
from langcodes.tag_parser import LanguageTagError, parse_tag, normalize_characters
from langcodes.language_distance import tuple_distance_cached
from langcodes.data_dicts import (
def assume_script(self) -> 'Language':
    """
        Fill in the script if it's missing, and if it can be assumed from the
        language subtag. This is the opposite of `simplify_script`.

        >>> Language.make(language='en').assume_script()
        Language.make(language='en', script='Latn')

        >>> Language.make(language='yi').assume_script()
        Language.make(language='yi', script='Hebr')

        >>> Language.make(language='yi', script='Latn').assume_script()
        Language.make(language='yi', script='Latn')

        This fills in nothing when the script cannot be assumed -- such as when
        the language has multiple scripts, or it has no standard orthography:

        >>> Language.make(language='sr').assume_script()
        Language.make(language='sr')

        >>> Language.make(language='eee').assume_script()
        Language.make(language='eee')

        It also dosn't fill anything in when the language is unspecified.

        >>> Language.make(territory='US').assume_script()
        Language.make(territory='US')
        """
    if self._assumed is not None:
        return self._assumed
    if self.language and (not self.script):
        try:
            self._assumed = self.update_dict({'script': DEFAULT_SCRIPTS[self.language]})
        except KeyError:
            self._assumed = self
    else:
        self._assumed = self
    return self._assumed