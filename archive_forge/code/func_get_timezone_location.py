from __future__ import annotations
import re
import warnings
from functools import lru_cache
from typing import TYPE_CHECKING, SupportsInt
import datetime
from collections.abc import Iterable
from babel import localtime
from babel.core import Locale, default_locale, get_global
from babel.localedata import LocaleDataDict
def get_timezone_location(dt_or_tzinfo: _DtOrTzinfo=None, locale: Locale | str | None=LC_TIME, return_city: bool=False) -> str:
    """Return a representation of the given timezone using "location format".

    The result depends on both the local display name of the country and the
    city associated with the time zone:

    >>> tz = get_timezone('America/St_Johns')
    >>> print(get_timezone_location(tz, locale='de_DE'))
    Kanada (St. John’s) (Ortszeit)
    >>> print(get_timezone_location(tz, locale='en'))
    Canada (St. John’s) Time
    >>> print(get_timezone_location(tz, locale='en', return_city=True))
    St. John’s
    >>> tz = get_timezone('America/Mexico_City')
    >>> get_timezone_location(tz, locale='de_DE')
    u'Mexiko (Mexiko-Stadt) (Ortszeit)'

    If the timezone is associated with a country that uses only a single
    timezone, just the localized country name is returned:

    >>> tz = get_timezone('Europe/Berlin')
    >>> get_timezone_name(tz, locale='de_DE')
    u'Mitteleurop\\xe4ische Zeit'

    .. versionadded:: 0.9

    :param dt_or_tzinfo: the ``datetime`` or ``tzinfo`` object that determines
                         the timezone; if `None`, the current date and time in
                         UTC is assumed
    :param locale: the `Locale` object, or a locale string
    :param return_city: True or False, if True then return exemplar city (location)
                        for the time zone
    :return: the localized timezone name using location format

    """
    locale = Locale.parse(locale)
    zone = _get_tz_name(dt_or_tzinfo)
    zone = get_global('zone_aliases').get(zone, zone)
    info = locale.time_zones.get(zone, {})
    region_format = locale.zone_formats['region']
    territory = get_global('zone_territories').get(zone)
    if territory not in locale.territories:
        territory = 'ZZ'
    territory_name = locale.territories[territory]
    if not return_city and territory and (len(get_global('territory_zones').get(territory, [])) == 1):
        return region_format % territory_name
    fallback_format = locale.zone_formats['fallback']
    if 'city' in info:
        city_name = info['city']
    else:
        metazone = get_global('meta_zones').get(zone)
        metazone_info = locale.meta_zones.get(metazone, {})
        if 'city' in metazone_info:
            city_name = metazone_info['city']
        elif '/' in zone:
            city_name = zone.split('/', 1)[1].replace('_', ' ')
        else:
            city_name = zone.replace('_', ' ')
    if return_city:
        return city_name
    return region_format % (fallback_format % {'0': city_name, '1': territory_name})