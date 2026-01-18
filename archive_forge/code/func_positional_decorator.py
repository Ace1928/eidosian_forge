import base64
import functools
import inspect
import json
import logging
import os
import warnings
import six
from six.moves import urllib
def positional_decorator(wrapped):

    @functools.wraps(wrapped)
    def positional_wrapper(*args, **kwargs):
        if len(args) > max_positional_args:
            plural_s = ''
            if max_positional_args != 1:
                plural_s = 's'
            message = '{function}() takes at most {args_max} positional argument{plural} ({args_given} given)'.format(function=wrapped.__name__, args_max=max_positional_args, args_given=len(args), plural=plural_s)
            if positional_parameters_enforcement == POSITIONAL_EXCEPTION:
                raise TypeError(message)
            elif positional_parameters_enforcement == POSITIONAL_WARNING:
                logger.warning(message)
        return wrapped(*args, **kwargs)
    return positional_wrapper