from __future__ import print_function
import re
import six
import numpy as np
from patsy import PatsyError
from patsy.origin import Origin
from patsy.util import (atleast_2d_column_default,
from patsy.infix_parser import Token, Operator, infix_parse
from patsy.parse_formula import _parsing_error_test
def linear_constraint(constraint_like, variable_names):
    """This is the internal interface implementing
    DesignInfo.linear_constraint, see there for docs."""
    if isinstance(constraint_like, LinearConstraint):
        if constraint_like.variable_names != variable_names:
            raise ValueError('LinearConstraint has wrong variable_names (got %r, expected %r)' % (constraint_like.variable_names, variable_names))
        return constraint_like
    if isinstance(constraint_like, Mapping):
        coefs = np.zeros((len(constraint_like), len(variable_names)), dtype=float)
        constants = np.zeros(len(constraint_like))
        used = set()
        for i, (name, value) in enumerate(six.iteritems(constraint_like)):
            if name in variable_names:
                idx = variable_names.index(name)
            elif isinstance(name, six.integer_types):
                idx = name
            else:
                raise ValueError('unrecognized variable name/index %r' % (name,))
            if idx in used:
                raise ValueError('duplicated constraint on %r' % (variable_names[idx],))
            used.add(idx)
            coefs[i, idx] = 1
            constants[i] = value
        return LinearConstraint(variable_names, coefs, constants)
    if isinstance(constraint_like, str):
        constraint_like = [constraint_like]
    if isinstance(constraint_like, list) and constraint_like and isinstance(constraint_like[0], str):
        constraints = []
        for code in constraint_like:
            if not isinstance(code, str):
                raise ValueError('expected a string, not %r' % (code,))
            tree = parse_constraint(code, variable_names)
            evaluator = _EvalConstraint(variable_names)
            constraints.append(evaluator.eval(tree, constraint=True))
        return LinearConstraint.combine(constraints)
    if isinstance(constraint_like, tuple):
        if len(constraint_like) != 2:
            raise ValueError('constraint tuple must have length 2')
        coef, constants = constraint_like
        return LinearConstraint(variable_names, coef, constants)
    coefs = np.asarray(constraint_like, dtype=float)
    return LinearConstraint(variable_names, coefs)