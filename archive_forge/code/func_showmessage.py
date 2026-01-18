from contextlib import contextmanager
import logging
import typing
import os
from rpy2.rinterface_lib import openrlib
from rpy2.rinterface_lib import ffi_proxy
from rpy2.rinterface_lib import conversion
def showmessage(s: str) -> None:
    print('R wants to show a message')
    print(s)