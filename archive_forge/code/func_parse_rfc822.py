from __future__ import annotations
import datetime
def parse_rfc822(date: str) -> datetime.datetime | None:
    """Parse RFC 822 dates and times.

    https://tools.ietf.org/html/rfc822#section-5

    The basic format is:

    ..  code-block:: text

        [ day "," ] dd mmm [yy]yy hh:mm[:ss] zzz

    Note that RFC 822 only specifies an explicit comma,
    but fails to make whitespace mandatory.

    Some non-standard formatting differences are allowed:

    *   Whitespace is assumed to separate each part of the timestamp.
    *   Years may be two or four digits.
        This is explicitly allowed in the OPML specification.
    *   The month name and day can be swapped.
    *   Timezones may be prefixed with "Etc/".
    *   If the time and/or timezone are missing,
        midnight and GMT will be assumed.
    *   "UTC" is supported as a timezone name.

    """
    parts = date.rpartition(',')[2].lower().split()
    if len(parts) == 3:
        parts.append('00:00:00')
    if len(parts) == 4:
        parts.append('gmt')
    elif len(parts) != 5:
        return None
    try:
        day = int(parts[0])
    except ValueError:
        if months.get(parts[0][:3]):
            try:
                day = int(parts[1])
            except ValueError:
                return None
            else:
                parts[1] = parts[0]
        else:
            return None
    month = months.get(parts[1][:3])
    if month is None:
        return None
    try:
        year = int(parts[2])
    except ValueError:
        return None
    if year < 100:
        if year >= 90:
            year += 1900
        else:
            year += 2000
    time_parts = parts[3].split(':')
    time_parts += ['0'] * (3 - len(time_parts))
    try:
        hour, minute, second = map(int, time_parts)
    except ValueError:
        return None
    tz_min = 0
    timezone = parts[4]
    if timezone.startswith('etc/'):
        timezone = timezone[4:] or 'gmt'
    if timezone.startswith('gmt'):
        timezone = timezone[3:] or 'gmt'
    tz_hour = timezones.get(timezone)
    if tz_hour is None:
        try:
            tz_left, tz_right = timezone.split(':')
            tz_hour = int(tz_left)
            tz_min = int(tz_right)
        except ValueError:
            try:
                tz_hour = int(timezone[:-2])
                tz_min = int(timezone[-2:])
            except ValueError:
                return None
        if tz_hour < 0:
            tz_min = tz_min * -1
    try:
        return datetime.datetime(year, month, day, hour, minute, second, tzinfo=datetime.timezone(datetime.timedelta(minutes=tz_hour * 60 + tz_min)))
    except (ValueError, OverflowError):
        return None