import re
from datetime import date, timedelta
from isodate.isostrf import strftime, DATE_EXT_COMPLETE
from isodate.isoerror import ISO8601Error
def build_date_regexps(yeardigits=4, expanded=False):
    """
    Compile set of regular expressions to parse ISO dates. The expressions will
    be created only if they are not already in REGEX_CACHE.

    It is necessary to fix the number of year digits, else it is not possible
    to automatically distinguish between various ISO date formats.

    ISO 8601 allows more than 4 digit years, on prior agreement, but then a +/-
    sign is required (expanded format). To support +/- sign for 4 digit years,
    the expanded parameter needs to be set to True.
    """
    if yeardigits != 4:
        expanded = True
    if (yeardigits, expanded) not in DATE_REGEX_CACHE:
        cache_entry = []
        if expanded:
            sign = 1
        else:
            sign = 0
        cache_entry.append(re.compile('(?P<sign>[+-]){%d}(?P<year>[0-9]{%d})-(?P<month>[0-9]{2})-(?P<day>[0-9]{2})' % (sign, yeardigits)))
        cache_entry.append(re.compile('(?P<sign>[+-]){%d}(?P<year>[0-9]{%d})(?P<month>[0-9]{2})(?P<day>[0-9]{2})' % (sign, yeardigits)))
        cache_entry.append(re.compile('(?P<sign>[+-]){%d}(?P<year>[0-9]{%d})-W(?P<week>[0-9]{2})-(?P<day>[0-9]{1})' % (sign, yeardigits)))
        cache_entry.append(re.compile('(?P<sign>[+-]){%d}(?P<year>[0-9]{%d})W(?P<week>[0-9]{2})(?P<day>[0-9]{1})' % (sign, yeardigits)))
        cache_entry.append(re.compile('(?P<sign>[+-]){%d}(?P<year>[0-9]{%d})-(?P<day>[0-9]{3})' % (sign, yeardigits)))
        cache_entry.append(re.compile('(?P<sign>[+-]){%d}(?P<year>[0-9]{%d})(?P<day>[0-9]{3})' % (sign, yeardigits)))
        cache_entry.append(re.compile('(?P<sign>[+-]){%d}(?P<year>[0-9]{%d})-W(?P<week>[0-9]{2})' % (sign, yeardigits)))
        cache_entry.append(re.compile('(?P<sign>[+-]){%d}(?P<year>[0-9]{%d})W(?P<week>[0-9]{2})' % (sign, yeardigits)))
        cache_entry.append(re.compile('(?P<sign>[+-]){%d}(?P<year>[0-9]{%d})-(?P<month>[0-9]{2})' % (sign, yeardigits)))
        cache_entry.append(re.compile('(?P<sign>[+-]){%d}(?P<year>[0-9]{%d})(?P<month>[0-9]{2})' % (sign, yeardigits)))
        cache_entry.append(re.compile('(?P<sign>[+-]){%d}(?P<year>[0-9]{%d})' % (sign, yeardigits)))
        cache_entry.append(re.compile('(?P<sign>[+-]){%d}(?P<century>[0-9]{%d})' % (sign, yeardigits - 2)))
        DATE_REGEX_CACHE[yeardigits, expanded] = cache_entry
    return DATE_REGEX_CACHE[yeardigits, expanded]