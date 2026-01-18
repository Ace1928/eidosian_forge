import re
import unicodedata
def ordinal(number):
    """
    Return the suffix that should be added to a number to denote the position
    in an ordered sequence such as 1st, 2nd, 3rd, 4th.

    Examples::

        >>> ordinal(1)
        'st'
        >>> ordinal(2)
        'nd'
        >>> ordinal(1002)
        'nd'
        >>> ordinal(1003)
        'rd'
        >>> ordinal(-11)
        'th'
        >>> ordinal(-1021)
        'st'

    """
    number = abs(int(number))
    if number % 100 in (11, 12, 13):
        return 'th'
    else:
        return {1: 'st', 2: 'nd', 3: 'rd'}.get(number % 10, 'th')