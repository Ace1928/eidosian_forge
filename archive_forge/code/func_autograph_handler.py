from tensorflow.python.autograph.core import converter
from tensorflow.python.autograph.impl import api
from tensorflow.python.util import tf_decorator
def autograph_handler(*args, **kwargs):
    """Calls a converted version of original_func."""
    try:
        return api.converted_call(original_func, args, kwargs, options=converter.ConversionOptions(recursive=True, optional_features=autograph_options, user_requested=True))
    except Exception as e:
        if hasattr(e, 'ag_error_metadata'):
            raise e.ag_error_metadata.to_exception(e)
        else:
            raise