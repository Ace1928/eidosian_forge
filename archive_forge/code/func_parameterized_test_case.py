import functools
import types
def parameterized_test_case(cls):
    """Class decorator to process parameterized tests

    This allows for parameterization to be used for potentially any
    unittest compatible runner; including testr and py.test.
    """
    tests_to_remove = []
    tests_to_add = []
    for key, val in vars(cls).items():
        if key.startswith('test_') and val.__dict__.get('build_data'):
            to_remove, to_add = process_parameterized_function(name=key, func_obj=val, build_data=val.__dict__.get('build_data'))
            tests_to_remove.extend(to_remove)
            tests_to_add.extend(to_add)
    [setattr(cls, name, func) for name, func in tests_to_add]
    [delattr(cls, key) for key in tests_to_remove if hasattr(cls, key)]
    return cls