import collections
import itertools
import typing  # pylint: disable=unused-import (used in doctests)
from tensorflow.python.framework import _pywrap_python_api_dispatcher as _api_dispatcher
from tensorflow.python.framework import ops
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_export as tf_export_lib
from tensorflow.python.util import tf_inspect
from tensorflow.python.util import traceback_utils
from tensorflow.python.util import type_annotations
from tensorflow.python.util.tf_export import tf_export
def make_type_checker(annotation):
    """Builds a PyTypeChecker for the given type annotation."""
    if type_annotations.is_generic_union(annotation):
        type_args = type_annotations.get_generic_type_args(annotation)
        simple_types = [t for t in type_args if isinstance(t, type)]
        simple_types = tuple(sorted(simple_types, key=id))
        if len(simple_types) > 1:
            if simple_types not in _is_instance_checker_cache:
                checker = _api_dispatcher.MakeInstanceChecker(*simple_types)
                _is_instance_checker_cache[simple_types] = checker
            options = [_is_instance_checker_cache[simple_types]] + [make_type_checker(t) for t in type_args if not isinstance(t, type)]
            return _api_dispatcher.MakeUnionChecker(options)
        options = [make_type_checker(t) for t in type_args]
        return _api_dispatcher.MakeUnionChecker(options)
    elif type_annotations.is_generic_list(annotation):
        type_args = type_annotations.get_generic_type_args(annotation)
        if len(type_args) != 1:
            raise AssertionError('Expected List[...] to have a single type parameter')
        elt_type = make_type_checker(type_args[0])
        return _api_dispatcher.MakeListChecker(elt_type)
    elif isinstance(annotation, type):
        if annotation not in _is_instance_checker_cache:
            checker = _api_dispatcher.MakeInstanceChecker(annotation)
            _is_instance_checker_cache[annotation] = checker
        return _is_instance_checker_cache[annotation]
    elif annotation is None:
        return make_type_checker(type(None))
    else:
        raise ValueError(f'Type annotation {annotation} is not currently supported by dispatch.  Supported annotations: type objects,  List[...], and Union[...]')