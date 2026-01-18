import datetime
from typing import Any, Optional, cast
from dateutil.rrule import WEEKLY, rrule
from arrow.constants import (
def iso_to_gregorian(iso_year: int, iso_week: int, iso_day: int) -> datetime.date:
    """Converts an ISO week date into a datetime object.

    :param iso_year: the year
    :param iso_week: the week number, each year has either 52 or 53 weeks
    :param iso_day: the day numbered 1 through 7, beginning with Monday

    """
    if not 1 <= iso_week <= 53:
        raise ValueError('ISO Calendar week value must be between 1-53.')
    if not 1 <= iso_day <= 7:
        raise ValueError('ISO Calendar day value must be between 1-7')
    fourth_jan = datetime.date(iso_year, 1, 4)
    delta = datetime.timedelta(fourth_jan.isoweekday() - 1)
    year_start = fourth_jan - delta
    gregorian = year_start + datetime.timedelta(days=iso_day - 1, weeks=iso_week - 1)
    return gregorian