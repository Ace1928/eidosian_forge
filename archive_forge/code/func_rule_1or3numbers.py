from __future__ import absolute_import
import re
from decimal import Decimal
from functools import partial
from six.moves import range
def rule_1or3numbers(self, next_val_fn, token):
    numbers = []
    token = next_val_fn()
    number, token = self.rule_number(next_val_fn, token)
    numbers.append(number)
    number, token = self.rule_optional_number(next_val_fn, token)
    if number is not None:
        numbers.append(number)
        number, token = self.rule_number(next_val_fn, token)
        numbers.append(number)
    return (numbers, token)