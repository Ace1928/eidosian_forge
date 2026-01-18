import enum
import logging
import os
import sys
import typing
import warnings
from rpy2.rinterface_lib import openrlib
from rpy2.rinterface_lib import callbacks
def isready() -> bool:
    """Is the embedded R ready for use."""
    INITIALIZED = RPY_R_Status.INITIALIZED
    return bool(rpy2_embeddedR_isinitialized == INITIALIZED.value)