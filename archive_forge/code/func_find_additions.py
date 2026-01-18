from jedi import debug
from jedi import settings
from jedi.inference import recursion
from jedi.inference.base_value import ValueSet, NO_VALUES, HelperValueMixin, \
from jedi.inference.lazy_value import LazyKnownValues
from jedi.inference.helpers import infer_call_of_leaf
from jedi.inference.cache import inference_state_method_cache
def find_additions(context, arglist, add_name):
    params = list(arguments.TreeArguments(context.inference_state, context, arglist).unpack())
    result = set()
    if add_name in ['insert']:
        params = params[1:]
    if add_name in ['append', 'add', 'insert']:
        for key, lazy_value in params:
            result.add(lazy_value)
    elif add_name in ['extend', 'update']:
        for key, lazy_value in params:
            result |= set(lazy_value.infer().iterate())
    return result