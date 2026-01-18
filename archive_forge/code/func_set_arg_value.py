import copy
import functools
import re
import weakref
import numpy as np
import tensorflow.compat.v2 as tf
from keras.src import initializers
from keras.src.utils import io_utils
from tensorflow.python.util.tf_export import keras_export
def set_arg_value(self, arg_name, new_value, args, kwargs, inputs_in_args=False, pop_kwarg_if_none=False):
    """Sets the value of an argument into the given args/kwargs.

        Args:
          arg_name: String name of the argument to find.
          new_value: New value to give to the argument.
          args: Tuple of args passed to the call function.
          kwargs: Dictionary of kwargs  passed to the call function.
          inputs_in_args: Whether the input argument (the first argument in the
            call function) is included in `args`. Defaults to `False`.
          pop_kwarg_if_none: If the new value is `None`, and this is `True`,
            then the argument is deleted from `kwargs`.

        Returns:
          The updated `(args, kwargs)`.
        """
    if self.full_argspec.varargs:
        try:
            arg_pos = self.full_argspec.args.index(arg_name)
            if self.full_argspec.args[0] == 'self':
                arg_pos -= 1
        except ValueError:
            arg_pos = None
    else:
        arg_pos = self.arg_positions.get(arg_name, None)
    if arg_pos is not None:
        if not inputs_in_args:
            arg_pos = arg_pos - 1
        if len(args) > arg_pos:
            args = list(args)
            args[arg_pos] = new_value
            return (tuple(args), kwargs)
    if new_value is None and pop_kwarg_if_none:
        kwargs.pop(arg_name, None)
    else:
        kwargs[arg_name] = new_value
    return (args, kwargs)