from operator import itemgetter
from typing import Any, List, Tuple, Dict, Sequence, Iterable, Optional, Mapping, Union
import warnings
import sys
from langcodes.tag_parser import LanguageTagError, parse_tag, normalize_characters
from langcodes.language_distance import tuple_distance_cached
from langcodes.data_dicts import (
def speaking_population(self) -> int:
    """
        Get an estimate of how many people in the world speak this language,
        derived from CLDR data. Requires that `language_data` is installed.

        Only the language and territory codes will be considered. If a
        territory code is included, the population will count only the
        speakers of the language in that territory.

        Script subtags are disregarded, because it doesn't make sense to ask
        how many people speak in a particular writing script.

        >>> Language.get('es').speaking_population()
        487664083
        >>> Language.get('pt').speaking_population()
        237135429
        >>> Language.get('es-BR').speaking_population()
        76218
        >>> Language.get('pt-BR').speaking_population()
        192661560
        >>> Language.get('vo').speaking_population()
        0
        """
    try:
        from language_data.population_data import LANGUAGE_SPEAKING_POPULATION
    except ImportError:
        print(LANGUAGE_NAME_IMPORT_MESSAGE, file=sys.stdout)
        raise
    lang = self._filter_attributes(['language', 'territory'])
    return LANGUAGE_SPEAKING_POPULATION.get(str(lang), 0)