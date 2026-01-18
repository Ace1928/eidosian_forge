import calendar
import re
import sys
from datetime import date
from datetime import datetime as dt_datetime
from datetime import time as dt_time
from datetime import timedelta
from datetime import tzinfo as dt_tzinfo
from math import trunc
from time import struct_time
from typing import (
from dateutil import tz as dateutil_tz
from dateutil.relativedelta import relativedelta
from arrow import formatter, locales, parser, util
from arrow.constants import DEFAULT_LOCALE, DEHUMANIZE_LOCALES
from arrow.locales import TimeFrameLiteral
def dehumanize(self, input_string: str, locale: str='en_us') -> 'Arrow':
    """Returns a new :class:`Arrow <arrow.arrow.Arrow>` object, that represents
        the time difference relative to the attributes of the
        :class:`Arrow <arrow.arrow.Arrow>` object.

        :param timestring: a ``str`` representing a humanized relative time.
        :param locale: (optional) a ``str`` specifying a locale.  Defaults to 'en-us'.

        Usage::

                >>> arw = arrow.utcnow()
                >>> arw
                <Arrow [2021-04-20T22:27:34.787885+00:00]>
                >>> earlier = arw.dehumanize("2 days ago")
                >>> earlier
                <Arrow [2021-04-18T22:27:34.787885+00:00]>

                >>> arw = arrow.utcnow()
                >>> arw
                <Arrow [2021-04-20T22:27:34.787885+00:00]>
                >>> later = arw.dehumanize("in a month")
                >>> later
                <Arrow [2021-05-18T22:27:34.787885+00:00]>

        """
    locale_obj = locales.get_locale(locale)
    normalized_locale_name = locale.lower().replace('_', '-')
    if normalized_locale_name not in DEHUMANIZE_LOCALES:
        raise ValueError(f'Dehumanize does not currently support the {locale} locale, please consider making a contribution to add support for this locale.')
    current_time = self.fromdatetime(self._datetime)
    time_object_info = dict.fromkeys(['seconds', 'minutes', 'hours', 'days', 'weeks', 'months', 'years'], 0)
    unit_visited = dict.fromkeys(['now', 'seconds', 'minutes', 'hours', 'days', 'weeks', 'months', 'years'], False)
    num_pattern = re.compile('\\d+')
    for unit, unit_object in locale_obj.timeframes.items():
        if isinstance(unit_object, Mapping):
            strings_to_search = unit_object
        else:
            strings_to_search = {unit: str(unit_object)}
        for time_delta, time_string in strings_to_search.items():
            search_string = str(time_string)
            search_string = search_string.format('\\d+')
            pattern = re.compile(f'(^|\\b|\\d){search_string}')
            match = pattern.search(input_string)
            if not match:
                continue
            match_string = match.group()
            num_match = num_pattern.search(match_string)
            if not num_match:
                change_value = 1 if not time_delta.isnumeric() else abs(int(time_delta))
            else:
                change_value = int(num_match.group())
            if unit == 'now':
                unit_visited[unit] = True
                continue
            time_unit_to_change = str(unit)
            time_unit_to_change += 's' if str(time_unit_to_change)[-1] != 's' else ''
            time_object_info[time_unit_to_change] = change_value
            unit_visited[time_unit_to_change] = True
    if not any([True for k, v in unit_visited.items() if v]):
        raise ValueError('Input string not valid. Note: Some locales do not support the week granularity in Arrow. If you are attempting to use the week granularity on an unsupported locale, this could be the cause of this error.')
    future_string = locale_obj.future
    future_string = future_string.format('.*')
    future_pattern = re.compile(f'^{future_string}$')
    future_pattern_match = future_pattern.findall(input_string)
    past_string = locale_obj.past
    past_string = past_string.format('.*')
    past_pattern = re.compile(f'^{past_string}$')
    past_pattern_match = past_pattern.findall(input_string)
    if past_pattern_match:
        sign_val = -1
    elif future_pattern_match:
        sign_val = 1
    elif unit_visited['now']:
        sign_val = 0
    else:
        raise ValueError("Invalid input String. String does not contain any relative time information. String should either represent a time in the future or a time in the past. Ex: 'in 5 seconds' or '5 seconds ago'.")
    time_changes = {k: sign_val * v for k, v in time_object_info.items()}
    return current_time.shift(**time_changes)