import functools
import types
def process_parameterized_function(name, func_obj, build_data):
    """Build lists of functions to add and remove to a test case."""
    to_remove = []
    to_add = []
    for subtest_name, params in build_data.items():
        func_name = '{0}_{1}'.format(name, subtest_name)
        new_func = construct_new_test_function(func_obj, func_name, params)
        to_add.append((func_name, new_func))
        to_remove.append(name)
    return (to_remove, to_add)