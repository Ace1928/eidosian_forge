from tensorflow.python.eager import def_function
from tensorflow.python.keras.saving.saved_model import constants
from tensorflow.python.keras.saving.saved_model import save_impl
from tensorflow.python.keras.utils.generic_utils import LazyLoader
from tensorflow.python.trackable import base as trackable
from tensorflow.python.trackable.autotrackable import AutoTrackable
def set_and_validate_functions(self, function_dict):
    """Saves function dictionary, and validates dictionary values."""
    for key in self.all_functions:
        if key in function_dict:
            if function_dict[key] is not None and (not isinstance(function_dict[key], (def_function.Function, save_impl.LayerCall))):
                raise ValueError('Function dictionary contained a non-function object: {} (for key {})'.format(function_dict[key], key))
            fn = function_dict[key]
            self._function_dict[key] = fn
            tf_fn = fn.wrapped_call if isinstance(fn, save_impl.LayerCall) else fn
            setattr(self._keras_trackable, key, tf_fn)
        else:
            raise ValueError('Function {} missing from serialized function dict.'.format(key))
    return self.functions