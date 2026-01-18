from contextlib import ExitStack
from functools import wraps, total_ordering
from inspect import getfullargspec as getargspec
import logging
import os
import re
import threading
import warnings
import attr
from rasterio._env import (
from rasterio._version import gdal_version
from rasterio.errors import EnvError, GDALVersionError, RasterioDeprecationWarning
from rasterio.session import Session, DummySession
def ensure_env_with_credentials(f):
    """Ensures a config environment exists and is credentialized

    Parameters
    ----------
    f : function
        A function.

    Returns
    -------
    A function wrapper.

    Notes
    -----
    The function wrapper checks the first argument of f and
    credentializes the environment if the first argument is a URI with
    scheme "s3".

    """

    @wraps(f)
    def wrapper(*args, **kwds):
        if local._env:
            env_ctor = Env
        else:
            env_ctor = Env.from_defaults
        fp_arg = kwds.get('fp', None) or args[0]
        if isinstance(fp_arg, str):
            session_cls = Session.cls_from_path(fp_arg)
            if local._env and session_cls.hascreds(getenv()):
                session_cls = DummySession
            session = session_cls()
        else:
            session = DummySession()
        with env_ctor(session=session):
            return f(*args, **kwds)
    return wrapper