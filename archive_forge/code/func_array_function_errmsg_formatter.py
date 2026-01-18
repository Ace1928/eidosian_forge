import ast
import re
import sys
import warnings
from ..exceptions import DTypePromotionError
from .multiarray import dtype, array, ndarray, promote_types
def array_function_errmsg_formatter(public_api, types):
    """ Format the error message for when __array_ufunc__ gives up. """
    func_name = '{}.{}'.format(public_api.__module__, public_api.__name__)
    return "no implementation found for '{}' on types that implement __array_function__: {}".format(func_name, list(types))