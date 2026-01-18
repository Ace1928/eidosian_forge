from __future__ import absolute_import, unicode_literals
import re
import sys
import unicodedata
import six
from pybtex.style.labels import BaseLabelStyle
from pybtex.textutils import abbreviate
def format_lab_names(self, persons):
    numnames = len(persons)
    if numnames > 1:
        if numnames > 4:
            namesleft = 3
        else:
            namesleft = numnames
        result = ''
        nameptr = 1
        while namesleft:
            person = persons[nameptr - 1]
            if nameptr == numnames:
                if six.text_type(person) == 'others':
                    result += '+'
                else:
                    result += _strip_nonalnum(_abbr(person.prelast_names + person.last_names))
            else:
                result += _strip_nonalnum(_abbr(person.prelast_names + person.last_names))
            nameptr += 1
            namesleft -= 1
        if numnames > 4:
            result += '+'
    else:
        person = persons[0]
        result = _strip_nonalnum(_abbr(person.prelast_names + person.last_names))
        if len(result) < 2:
            result = _strip_nonalnum(person.last_names)[:3]
    return result