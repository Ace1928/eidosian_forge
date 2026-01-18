import enum
import logging
import os
import sys
import typing
import warnings
from rpy2.rinterface_lib import openrlib
from rpy2.rinterface_lib import callbacks
def assert_isready() -> None:
    """Assert whether R is ready (initialized).

    Raises an RNotReadyError if it is not."""
    if not isready():
        raise RNotReadyError('The embedded R is not ready to use.')