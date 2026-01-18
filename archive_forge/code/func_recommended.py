import configparser
import os
import sys
import logging
import typing
from . import backend, credentials
from .util import platform_ as platform
from .backends import fail
def recommended(backend) -> bool:
    return backend.priority >= 1