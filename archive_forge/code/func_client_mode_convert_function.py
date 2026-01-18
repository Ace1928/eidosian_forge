import os
import threading
from contextlib import contextmanager
from functools import wraps
from ray._private.auto_init_hook import auto_init_ray
def client_mode_convert_function(func_cls, in_args, in_kwargs, **kwargs):
    """Runs a preregistered ray RemoteFunction through the ray client.

    The common case for this is to transparently convert that RemoteFunction
    to a ClientRemoteFunction. This happens in circumstances where the
    RemoteFunction is declared early, in a library and only then is Ray used in
    client mode -- necessitating a conversion.
    """
    from ray.util.client import ray
    key = getattr(func_cls, RAY_CLIENT_MODE_ATTR, None)
    if key is None or not ray._converted_key_exists(key):
        key = ray._convert_function(func_cls)
        setattr(func_cls, RAY_CLIENT_MODE_ATTR, key)
    client_func = ray._get_converted(key)
    return client_func._remote(in_args, in_kwargs, **kwargs)