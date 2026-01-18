import pytest
from nipype.utils.functions import getsource, create_function_from_source
def test_func_string():

    def is_string():
        return isinstance('string', str)
    wrapped_func = create_function_from_source(getsource(is_string))
    assert is_string() == wrapped_func()