import calendar
import datetime
import decimal
import re
from typing import Any, Iterator, List, Optional, Tuple, Union
from unicodedata import category
from ..exceptions import xpath_error
from ..regex import translate_pattern
from ._translation_maps import ALPHABET_CHARACTERS, OTHER_NUMBERS, ROMAN_NUMERALS_MAP, \
def parse_datetime_picture(picture: str) -> Tuple[List[str], List[str]]:
    """
    Analyze a picture argument of XPath 3.0+ formatting functions.

    :param picture: the picture string.
    :return: a couple of lists containing the literal parts and markers.
    """
    min_value: Union[int, str]
    max_value: Union[None, int, str]
    literals = []
    for lit in PICTURE_PATTERN.split(picture):
        if '[' in lit.replace('[[', ''):
            raise xpath_error('FOFD1340', "Invalid character '[' in picture literal")
        elif ']' in lit.replace(']]', ''):
            raise xpath_error('FOFD1340', "Invalid character ']' in picture literal")
        else:
            literals.append(lit.replace('[[', '[').replace(']]', ']'))
    markers = [x.group().replace(' ', '').replace('\n', '').replace('\t', '') for x in PICTURE_PATTERN.finditer(picture)]
    assert len(markers) == len(literals) - 1
    msg_tmpl = 'Invalid formatting component {!r}'
    for value in markers:
        if value[1] not in 'YMDdFWwHhPmsfZzCE':
            raise xpath_error('FOFD1340', msg_tmpl.format(value))
        if ',' not in value:
            presentation = value[2:-1]
        else:
            presentation, width = value[2:-1].rsplit(',', maxsplit=1)
            if WIDTH_PATTERN.match(width) is None:
                raise xpath_error('FOFD1340', f'Invalid width modifier {value!r}')
            elif '-' not in width:
                if '*' not in width and (not int(width)):
                    raise xpath_error('FOFD1340', f'Invalid width modifier {value!r}')
            elif '*' not in width:
                min_value, max_value = map(int, width.split('-'))
                if min_value < 1 or max_value < min_value:
                    raise xpath_error('FOFD1340', msg_tmpl.format(value))
            else:
                min_value, max_value = width.split('-')
                if min_value != '*' and (not int(min_value)):
                    raise xpath_error('FOFD1340', f'Invalid width modifier {value!r}')
                if max_value != '*' and (not int(max_value)):
                    raise xpath_error('FOFD1340', f'Invalid width modifier {value!r}')
        if len(presentation) > 1 and presentation[-1] in 'atco':
            presentation = presentation[:-1]
        if not presentation or presentation in PRESENTATION_FORMATS:
            pass
        elif DECIMAL_DIGIT_PATTERN.match(presentation) is None:
            raise xpath_error('FOFD1340', msg_tmpl.format(value))
        else:
            if value[1] == 'f':
                if presentation[0] == '#' and any((ch.isdigit() for ch in presentation)):
                    msg = 'picture argument has an invalid primary format token'
                    raise xpath_error('FOFD1340', msg)
            elif presentation[0].isdigit() and '#' in presentation:
                msg = 'picture argument has an invalid primary format token'
                raise xpath_error('FOFD1340', msg)
            cp = None
            for ch in reversed(presentation):
                if not ch.isdigit():
                    continue
                elif cp is None:
                    cp = ord(ch)
                elif abs(ord(ch) - cp) > 10:
                    raise xpath_error('FOFD1340', msg_tmpl.format(value))
    return (literals, markers)