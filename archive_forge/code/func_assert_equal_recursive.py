from minerl.herobraine.hero.spaces import Box, Dict, Discrete, MultiDiscrete, Enum
import collections
import numpy as np
def assert_equal_recursive(npa_dict, dict_to_test, atol=1e-08, ignore=None):
    ignore = [] if ignore is None else ignore
    assert isinstance(npa_dict, collections.OrderedDict)
    assert isinstance(dict_to_test, collections.OrderedDict)
    for key, value in npa_dict.items():
        if key in ignore:
            continue
        if isinstance(value, np.ndarray):
            if key == 'camera':
                assert np.allclose(value, dict_to_test[key], atol=1.5)
            elif value.dtype.type is np.string_ or value.dtype.type is np.str_:
                assert value == dict_to_test[key]
            else:
                assert np.allclose(value, dict_to_test[key], atol=atol)
        elif isinstance(value, collections.OrderedDict):
            assert_equal_recursive(value, dict_to_test[key], atol, ignore)
        else:
            assert value == dict_to_test[key]