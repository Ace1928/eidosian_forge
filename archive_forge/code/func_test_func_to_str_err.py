import pytest
from nipype.utils.functions import getsource, create_function_from_source
def test_func_to_str_err():
    bad_src = 'obbledygobbledygook'
    with pytest.raises(RuntimeError):
        create_function_from_source(bad_src)