from __future__ import absolute_import
import re
from decimal import Decimal, getcontext
from functools import partial
def rule_orthogonal_lineto(self, next_val_fn, token):
    command = token[1]
    token = next_val_fn()
    coordinates = []
    while token[0] in self.number_tokens:
        coord, token = self.rule_coordinate(next_val_fn, token)
        coordinates.append(coord)
    return ((command, coordinates), token)