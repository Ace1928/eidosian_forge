import builtins
import time
from concurrent.futures import ThreadPoolExecutor
import pytest
import sklearn
from sklearn import config_context, get_config, set_config
from sklearn.utils import _IS_WASM
from sklearn.utils.parallel import Parallel, delayed
def test_config_context():
    assert get_config() == {'assume_finite': False, 'working_memory': 1024, 'print_changed_only': True, 'display': 'diagram', 'array_api_dispatch': False, 'pairwise_dist_chunk_size': 256, 'enable_cython_pairwise_dist': True, 'transform_output': 'default', 'enable_metadata_routing': False, 'skip_parameter_validation': False}
    config_context(assume_finite=True)
    assert get_config()['assume_finite'] is False
    with config_context(assume_finite=True):
        assert get_config() == {'assume_finite': True, 'working_memory': 1024, 'print_changed_only': True, 'display': 'diagram', 'array_api_dispatch': False, 'pairwise_dist_chunk_size': 256, 'enable_cython_pairwise_dist': True, 'transform_output': 'default', 'enable_metadata_routing': False, 'skip_parameter_validation': False}
    assert get_config()['assume_finite'] is False
    with config_context(assume_finite=True):
        with config_context(assume_finite=None):
            assert get_config()['assume_finite'] is True
        assert get_config()['assume_finite'] is True
        with config_context(assume_finite=False):
            assert get_config()['assume_finite'] is False
            with config_context(assume_finite=None):
                assert get_config()['assume_finite'] is False
                set_config(assume_finite=True)
                assert get_config()['assume_finite'] is True
            assert get_config()['assume_finite'] is False
        assert get_config()['assume_finite'] is True
    assert get_config() == {'assume_finite': False, 'working_memory': 1024, 'print_changed_only': True, 'display': 'diagram', 'array_api_dispatch': False, 'pairwise_dist_chunk_size': 256, 'enable_cython_pairwise_dist': True, 'transform_output': 'default', 'enable_metadata_routing': False, 'skip_parameter_validation': False}
    with pytest.raises(TypeError):
        config_context(True)
    with pytest.raises(TypeError):
        config_context(do_something_else=True).__enter__()