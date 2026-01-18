from __future__ import absolute_import
import re
from decimal import Decimal, getcontext
from functools import partial
def rule_curveto2(self, next_val_fn, token):
    command = token[1]
    token = next_val_fn()
    coordinates = []
    while token[0] in self.number_tokens:
        pair1, token = self.rule_coordinate_pair(next_val_fn, token)
        pair2, token = self.rule_coordinate_pair(next_val_fn, token)
        coordinates.extend(pair1)
        coordinates.extend(pair2)
    return ((command, coordinates), token)