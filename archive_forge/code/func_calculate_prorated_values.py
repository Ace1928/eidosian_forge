import datetime
import time
import re
import numbers
import functools
import contextlib
from numbers import Number
from typing import Union, Tuple, Iterable
from typing import cast
def calculate_prorated_values():
    """
    >>> monkeypatch = getfixture('monkeypatch')
    >>> import builtins
    >>> monkeypatch.setattr(builtins, 'input', lambda prompt: '3/hour')
    >>> calculate_prorated_values()
    per minute: 0.05
    per hour: 3.0
    per day: 72.0
    per month: 2191.454166666667
    per year: 26297.45
    """
    rate = input('Enter the rate (3/hour, 50/month)> ')
    for period, value in _prorated_values(rate):
        print(f'per {period}: {value}')