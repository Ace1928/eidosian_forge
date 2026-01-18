import pytest
from nipype.utils.functions import getsource, create_function_from_source
def test_func_print():
    wrapped_func = create_function_from_source(getsource(_print_statement))
    assert wrapped_func()