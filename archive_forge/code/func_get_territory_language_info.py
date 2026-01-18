from __future__ import annotations
from babel.core import get_global
def get_territory_language_info(territory: str) -> dict[str, dict[str, float | str | None]]:
    """
    Get a dictionary of language information for a territory.

    The dictionary is keyed by language code; the values are dicts with more information.

    The following keys are currently known for the values:

    * `population_percent`: The percentage of the territory's population speaking the
                            language.
    * `official_status`: An optional string describing the officiality status of the language.
                         Known values are "official", "official_regional" and "de_facto_official".

    .. warning:: Note that the data is as up to date as the current version of the CLDR used
                 by Babel.  If you need scientifically accurate information, use another source!

    .. note:: Note that the format of the dict returned may change between Babel versions.

    See https://www.unicode.org/cldr/charts/latest/supplemental/territory_language_information.html

    :param territory: Territory code
    :type territory: str
    :return: Language information dictionary
    :rtype: dict[str, dict]
    """
    territory = str(territory).upper()
    return get_global('territory_languages').get(territory, {}).copy()