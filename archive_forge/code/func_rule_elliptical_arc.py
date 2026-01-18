from __future__ import absolute_import
import re
from decimal import Decimal, getcontext
from functools import partial
def rule_elliptical_arc(self, next_val_fn, token):
    command = token[1]
    token = next_val_fn()
    arguments = []
    while token[0] in self.number_tokens:
        rx = Decimal(token[1]) * 1
        if rx < Decimal('0.0'):
            raise SyntaxError('expecting a nonnegative number; got %r' % (token,))
        token = next_val_fn()
        if token[0] not in self.number_tokens:
            raise SyntaxError('expecting a number; got %r' % (token,))
        ry = Decimal(token[1]) * 1
        if ry < Decimal('0.0'):
            raise SyntaxError('expecting a nonnegative number; got %r' % (token,))
        token = next_val_fn()
        if token[0] not in self.number_tokens:
            raise SyntaxError('expecting a number; got %r' % (token,))
        axis_rotation = Decimal(token[1]) * 1
        token = next_val_fn()
        if token[1][0] not in ('0', '1'):
            raise SyntaxError('expecting a boolean flag; got %r' % (token,))
        large_arc_flag = Decimal(token[1][0]) * 1
        if len(token[1]) > 1:
            token = list(token)
            token[1] = token[1][1:]
        else:
            token = next_val_fn()
        if token[1][0] not in ('0', '1'):
            raise SyntaxError('expecting a boolean flag; got %r' % (token,))
        sweep_flag = Decimal(token[1][0]) * 1
        if len(token[1]) > 1:
            token = list(token)
            token[1] = token[1][1:]
        else:
            token = next_val_fn()
        if token[0] not in self.number_tokens:
            raise SyntaxError('expecting a number; got %r' % (token,))
        x = Decimal(token[1]) * 1
        token = next_val_fn()
        if token[0] not in self.number_tokens:
            raise SyntaxError('expecting a number; got %r' % (token,))
        y = Decimal(token[1]) * 1
        token = next_val_fn()
        arguments.extend([rx, ry, axis_rotation, large_arc_flag, sweep_flag, x, y])
    return ((command, arguments), token)