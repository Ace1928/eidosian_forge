import re
import traitlets
import datetime as dt
def naive_to_json(pydt, manager):
    """Serialize a naive Python datetime object to json.

    Instantiating a JavaScript Date object with a string assumes that the
    string is a UTC string, while instantiating it with constructor arguments
    assumes that it's in local time:

    >>> cdate = new Date('2015-05-12')
    Mon May 11 2015 20:00:00 GMT-0400 (Eastern Daylight Time)
    >>> cdate = new Date(2015, 4, 12) // Months are 0-based indices in JS
    Tue May 12 2015 00:00:00 GMT-0400 (Eastern Daylight Time)

    Attributes of this dictionary are to be passed to the JavaScript Date
    constructor.
    """
    if pydt is None:
        return None
    else:
        naivedt = pydt.replace(tzinfo=None)
        return dict(year=naivedt.year, month=naivedt.month - 1, date=naivedt.day, hours=naivedt.hour, minutes=naivedt.minute, seconds=naivedt.second, milliseconds=naivedt.microsecond / 1000)