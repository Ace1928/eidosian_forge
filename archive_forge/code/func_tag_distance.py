from operator import itemgetter
from typing import Any, List, Tuple, Dict, Sequence, Iterable, Optional, Mapping, Union
import warnings
import sys
from langcodes.tag_parser import LanguageTagError, parse_tag, normalize_characters
from langcodes.language_distance import tuple_distance_cached
from langcodes.data_dicts import (
def tag_distance(desired: Union[str, Language], supported: Union[str, Language]) -> int:
    """
    Tags that expand to the same thing when likely values are filled in get a
    distance of 0.

    >>> tag_distance('en', 'en')
    0
    >>> tag_distance('en', 'en-US')
    0
    >>> tag_distance('zh-Hant', 'zh-TW')
    0
    >>> tag_distance('ru-Cyrl', 'ru')
    0

    As a specific example, Serbo-Croatian is a politically contentious idea,
    but in CLDR, it's considered equivalent to Serbian in Latin characters.

    >>> tag_distance('sh', 'sr-Latn')
    0

    ... which is very similar to Croatian but sociopolitically not the same.

    >>> tag_distance('sh', 'hr')
    9

    Unicode reorganized its distinction between 'no' (Norwegian) and 'nb'
    (Norwegian Bokmål) in 2021. 'no' is preferred in most contexts, and the more
    specific 'nb' is a distance of 1 from it:

    >>> tag_distance('nb', 'no')
    1

    These distances can be asymmetrical: this data includes the fact that speakers
    of Swiss German (gsw) know High German (de), but not at all the other way around.

    The difference seems a little bit extreme, but the asymmetry is certainly
    there. And if your text is tagged as 'gsw', it must be that way for a
    reason.

    >>> tag_distance('gsw', 'de')
    8
    >>> tag_distance('de', 'gsw')
    84

    Unconnected languages get a distance of 80 to 134.

    >>> tag_distance('en', 'zh')
    134
    >>> tag_distance('es', 'fr')
    84
    >>> tag_distance('fr-CH', 'de-CH')
    80

    Different local variants of the same language get a distance from 3 to 5.
    >>> tag_distance('zh-HK', 'zh-MO')   # Chinese is similar in Hong Kong and Macao
    4
    >>> tag_distance('en-AU', 'en-GB')   # Australian English is similar to British English
    3
    >>> tag_distance('en-IN', 'en-GB')   # Indian English is also similar to British English
    3
    >>> tag_distance('es-PE', 'es-419')  # Peruvian Spanish is Latin American Spanish
    1
    >>> tag_distance('es-419', 'es-PE')  # but Latin American Spanish is not necessarily Peruvian
    4
    >>> tag_distance('es-ES', 'es-419')  # Spanish in Spain is further from Latin American Spanish
    5
    >>> tag_distance('en-US', 'en-GB')   # American and British English are somewhat different
    5
    >>> tag_distance('es-MX', 'es-ES')   # Mexican Spanish is different from Spanish Spanish
    5
    >>> # European Portuguese is different from the most common form (Brazilian Portuguese)
    >>> tag_distance('pt', 'pt-PT')
    5

    >>> # Serbian has two scripts, and people might prefer one but understand both
    >>> tag_distance('sr-Latn', 'sr-Cyrl')
    5

    A distance of 10 is used for matching a specific language to its
    more-commonly-used macrolanguage tag.

    >>> tag_distance('arz', 'ar')  # Egyptian Arabic to Modern Standard Arabic
    10
    >>> tag_distance('wuu', 'zh')  # Wu Chinese to (Mandarin) Chinese
    10

    Higher distances can arrive due to particularly contentious differences in
    the script for writing the language, where people who understand one script
    can learn the other but may not be happy with it. This specifically applies
    to Chinese.

    >>> tag_distance('zh-TW', 'zh-CN')
    54
    >>> tag_distance('zh-Hans', 'zh-Hant')
    54
    >>> tag_distance('zh-CN', 'zh-HK')
    54
    >>> tag_distance('zh-CN', 'zh-TW')
    54
    >>> tag_distance('zh-Hant', 'zh-Hans')
    54

    This distance range also applies to the differences between Norwegian
    Bokmål, Nynorsk, and Danish.

    >>> tag_distance('no', 'da')
    12
    >>> tag_distance('no', 'nn')
    20

    Differences of 20 to 50 can represent substantially different languages,
    in cases where speakers of the first may understand the second for demographic
    reasons.

    >>> tag_distance('eu', 'es')  # Basque to Spanish
    20
    >>> tag_distance('af', 'nl')  # Afrikaans to Dutch
    24
    >>> tag_distance('mr', 'hi')  # Marathi to Hindi
    30
    >>> tag_distance('ms', 'id')  # Malay to Indonesian
    34
    >>> tag_distance('mg', 'fr')  # Malagasy to French
    34
    >>> tag_distance('ta', 'en')  # Tamil to English
    44

    A complex example is the tag 'yue' for Cantonese. Written Chinese is usually
    presumed to be Mandarin Chinese, but colloquial Cantonese can be written as
    well. (Some things could not be written any other way, such as Cantonese
    song lyrics.)

    The difference between Cantonese and Mandarin also implies script and
    territory differences by default, adding to the distance.

    >>> tag_distance('yue', 'zh')
    64

    When the supported script is a different one than desired, this is usually
    a major difference with score of 50 or more.

    >>> tag_distance('ja', 'ja-Latn-US-hepburn')
    54

    >>> # You can read the Shavian script, right?
    >>> tag_distance('en', 'en-Shaw')
    54
    """
    desired_obj = Language.get(desired)
    supported_obj = Language.get(supported)
    return desired_obj.distance(supported_obj)