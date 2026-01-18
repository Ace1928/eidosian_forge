import re
import inspect
import os
import sys
from importlib.machinery import SourceFileLoader
def initconf():
    """
    Initializes the default configuration and exposes it at
    ``pecan.configuration.conf``, which is also exposed at ``pecan.conf``.
    """
    return conf_from_dict(DEFAULT)