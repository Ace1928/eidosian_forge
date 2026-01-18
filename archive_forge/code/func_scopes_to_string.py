import base64
import functools
import inspect
import json
import logging
import os
import warnings
import six
from six.moves import urllib
def scopes_to_string(scopes):
    """Converts scope value to a string.

    If scopes is a string then it is simply passed through. If scopes is an
    iterable then a string is returned that is all the individual scopes
    concatenated with spaces.

    Args:
        scopes: string or iterable of strings, the scopes.

    Returns:
        The scopes formatted as a single string.
    """
    if isinstance(scopes, six.string_types):
        return scopes
    else:
        return ' '.join(scopes)