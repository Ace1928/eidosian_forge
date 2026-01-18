import re
from itertools import zip_longest
from parso.python import tree
from jedi import debug
from jedi.inference.utils import PushBackIterator
from jedi.inference import analysis
from jedi.inference.lazy_value import LazyKnownValue, LazyKnownValues, \
from jedi.inference.names import ParamName, TreeNameDefinition, AnonymousParamName
from jedi.inference.base_value import NO_VALUES, ValueSet, ContextualizedNode
from jedi.inference.value import iterable
from jedi.inference.cache import inference_state_as_method_param_cache
def iterate_argument_clinic(inference_state, arguments, clinic_string):
    """Uses a list with argument clinic information (see PEP 436)."""
    clinic_args = list(_parse_argument_clinic(clinic_string))
    iterator = PushBackIterator(arguments.unpack())
    for i, (name, optional, allow_kwargs, stars) in enumerate(clinic_args):
        if stars == 1:
            lazy_values = []
            for key, argument in iterator:
                if key is not None:
                    iterator.push_back((key, argument))
                    break
                lazy_values.append(argument)
            yield ValueSet([iterable.FakeTuple(inference_state, lazy_values)])
            lazy_values
            continue
        elif stars == 2:
            raise NotImplementedError()
        key, argument = next(iterator, (None, None))
        if key is not None:
            debug.warning('Keyword arguments in argument clinic are currently not supported.')
            raise ParamIssue
        if argument is None and (not optional):
            debug.warning('TypeError: %s expected at least %s arguments, got %s', name, len(clinic_args), i)
            raise ParamIssue
        value_set = NO_VALUES if argument is None else argument.infer()
        if not value_set and (not optional):
            debug.warning('argument_clinic "%s" not resolvable.', name)
            raise ParamIssue
        yield value_set