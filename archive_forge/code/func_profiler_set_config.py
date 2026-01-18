import ctypes
import warnings
from .base import _LIB, check_call, c_str, ProfileHandle, c_str_array, py_str, KVStoreHandle
def profiler_set_config(mode='symbolic', filename='profile.json'):
    """Set up the configure of profiler (Deprecated).

    Parameters
    ----------
    mode : string, optional
        Indicates whether to enable the profiler, can
        be 'symbolic', or 'all'. Defaults to `symbolic`.
    filename : string, optional
        The name of output trace file. Defaults to 'profile.json'.
    """
    warnings.warn('profiler.profiler_set_config() is deprecated. Please use profiler.set_config() instead')
    keys = c_str_array([key for key in ['profile_' + mode, 'filename']])
    values = c_str_array([str(val) for val in [True, filename]])
    assert len(keys) == len(values)
    check_call(_LIB.MXSetProcessProfilerConfig(len(keys), keys, values, profiler_kvstore_handle))