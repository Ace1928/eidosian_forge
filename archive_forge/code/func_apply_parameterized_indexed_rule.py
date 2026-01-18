import logging
import sys
from pyomo.common.deprecation import relocated_module_attribute
def apply_parameterized_indexed_rule(obj, rule, model, param, index):
    if index.__class__ is tuple:
        return rule(model, param, *index)
    if index is None:
        return rule(model, param)
    return rule(model, param, index)