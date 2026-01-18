from __future__ import absolute_import
import re
from decimal import Decimal
from functools import partial
from six.moves import range
def rule_1number(self, next_val_fn, token):
    token = next_val_fn()
    number, token = self.rule_number(next_val_fn, token)
    numbers = [number]
    return (numbers, token)