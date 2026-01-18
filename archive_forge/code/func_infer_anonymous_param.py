import sys
from typing import List
from pathlib import Path
from parso.tree import search_ancestor
from jedi.inference.cache import inference_state_method_cache
from jedi.inference.imports import goto_import, load_module_from_path
from jedi.inference.filters import ParserTreeFilter
from jedi.inference.base_value import NO_VALUES, ValueSet
from jedi.inference.helpers import infer_call_of_leaf
def infer_anonymous_param(func):

    def get_returns(value):
        if value.tree_node.annotation is not None:
            result = value.execute_with_values()
            if any((v.name.get_qualified_names(include_module_names=True) == ('typing', 'Generator') for v in result)):
                return ValueSet.from_sets((v.py__getattribute__('__next__').execute_annotation() for v in result))
            return result
        function_context = value.as_context()
        if function_context.is_generator():
            return function_context.merge_yield_values()
        else:
            return function_context.get_return_values()

    def wrapper(param_name):
        if param_name.annotation_node:
            return func(param_name)
        is_pytest_param, param_name_is_function_name = _is_a_pytest_param_and_inherited(param_name)
        if is_pytest_param:
            module = param_name.get_root_context()
            fixtures = _goto_pytest_fixture(module, param_name.string_name, skip_own_module=param_name_is_function_name)
            if fixtures:
                return ValueSet.from_sets((get_returns(value) for fixture in fixtures for value in fixture.infer()))
        return func(param_name)
    return wrapper