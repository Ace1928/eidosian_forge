import enum
import logging
import os
import sys
import typing
import warnings
from rpy2.rinterface_lib import openrlib
from rpy2.rinterface_lib import callbacks
def get_initoptions() -> typing.Tuple[str, ...]:
    """Get the initialization options for the embedded R."""
    return _options