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
def get_period_id(time: _Instant, tzinfo: datetime.tzinfo | None=None, type: Literal['selection'] | None=None, locale: Locale | str | None=LC_TIME) -> str:
    """
    Get the day period ID for a given time.

    This ID can be used as a key for the period name dictionary.

    >>> from datetime import time
    >>> get_period_names(locale="de")[get_period_id(time(7, 42), locale="de")]
    u'Morgen'

    >>> get_period_id(time(0), locale="en_US")
    u'midnight'

    >>> get_period_id(time(0), type="selection", locale="en_US")
    u'night1'

    :param time: The time to inspect.
    :param tzinfo: The timezone for the time. See ``format_time``.
    :param type: The period type to use. Either "selection" or None.
                 The selection type is used for selecting among phrases such as
                 “Your email arrived yesterday evening” or “Your email arrived last night”.
    :param locale: the `Locale` object, or a locale string
    :return: period ID. Something is always returned -- even if it's just "am" or "pm".
    """
    time = _get_time(time, tzinfo)
    seconds_past_midnight = int(time.hour * 60 * 60 + time.minute * 60 + time.second)
    locale = Locale.parse(locale)
    rulesets = locale.day_period_rules.get(type, {}).items()
    for rule_id, rules in rulesets:
        for rule in rules:
            if 'at' in rule and rule['at'] == seconds_past_midnight:
                return rule_id
    for rule_id, rules in rulesets:
        for rule in rules:
            if 'from' in rule and 'before' in rule:
                if rule['from'] < rule['before']:
                    if rule['from'] <= seconds_past_midnight < rule['before']:
                        return rule_id
                elif rule['from'] <= seconds_past_midnight < 86400 or 0 <= seconds_past_midnight < rule['before']:
                    return rule_id
            start_ok = end_ok = False
            if 'from' in rule and seconds_past_midnight >= rule['from']:
                start_ok = True
            if 'to' in rule and seconds_past_midnight <= rule['to']:
                end_ok = True
            if 'before' in rule and seconds_past_midnight < rule['before']:
                end_ok = True
            if 'after' in rule:
                raise NotImplementedError("'after' is deprecated as of CLDR 29.")
            if start_ok and end_ok:
                return rule_id
    if seconds_past_midnight < 43200:
        return 'am'
    else:
        return 'pm'