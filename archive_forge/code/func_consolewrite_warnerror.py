from contextlib import contextmanager
import logging
import typing
import os
from rpy2.rinterface_lib import openrlib
from rpy2.rinterface_lib import ffi_proxy
from rpy2.rinterface_lib import conversion
def consolewrite_warnerror(s: str) -> None:
    logger.warning(_WRITECONSOLE_EXCEPTION_LOG, s)