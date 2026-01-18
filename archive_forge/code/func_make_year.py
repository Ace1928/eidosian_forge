import cgi
import re
import warnings
from encodings import idna
from .api import (FancyValidator, Identity, Invalid, NoDefault, Validator,
def make_year(self, year, state):
    try:
        year = int(year)
    except ValueError:
        raise Invalid(self.message('invalidYear', state), year, state)
    if year <= 20:
        year += 2000
    elif 50 <= year < 100:
        year += 1900
    if 20 < year < 50 or 99 < year < 1900:
        raise Invalid(self.message('fourDigitYear', state), year, state)
    return year