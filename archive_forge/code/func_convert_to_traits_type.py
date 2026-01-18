import os.path as op
import inspect
import numpy as np
from ... import logging
from ..base import (
def convert_to_traits_type(dipy_type, is_file=False):
    """Convert DIPY type to Traits type."""
    dipy_type = dipy_type.lower()
    is_mandatory = bool('optional' not in dipy_type)
    if 'variable' in dipy_type and 'str' in dipy_type:
        return (traits.ListStr, is_mandatory)
    elif 'variable' in dipy_type and 'int' in dipy_type:
        return (traits.ListInt, is_mandatory)
    elif 'variable' in dipy_type and 'float' in dipy_type:
        return (traits.ListFloat, is_mandatory)
    elif 'variable' in dipy_type and 'bool' in dipy_type:
        return (traits.ListBool, is_mandatory)
    elif 'variable' in dipy_type and 'complex' in dipy_type:
        return (traits.ListComplex, is_mandatory)
    elif 'str' in dipy_type and (not is_file):
        return (traits.Str, is_mandatory)
    elif 'str' in dipy_type and is_file:
        return (File, is_mandatory)
    elif 'int' in dipy_type:
        return (traits.Int, is_mandatory)
    elif 'float' in dipy_type:
        return (traits.Float, is_mandatory)
    elif 'bool' in dipy_type:
        return (traits.Bool, is_mandatory)
    elif 'complex' in dipy_type:
        return (traits.Complex, is_mandatory)
    else:
        msg = 'Error during convert_to_traits_type({0}).'.format(dipy_type) + 'Unknown DIPY type.'
        raise IOError(msg)