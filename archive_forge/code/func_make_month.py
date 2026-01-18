import cgi
import re
import warnings
from encodings import idna
from .api import (FancyValidator, Identity, Invalid, NoDefault, Validator,
def make_month(self, value, state):
    try:
        return int(value)
    except ValueError:
        try:
            return self._month_names[value.lower().strip()]
        except KeyError:
            raise Invalid(self.message('unknownMonthName', state, month=value), value, state)