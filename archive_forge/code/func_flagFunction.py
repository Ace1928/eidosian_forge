from __future__ import annotations
import getopt
import inspect
import os
import sys
import textwrap
from os import path
from typing import Any, Dict, Optional, cast
from twisted.python import reflect, util
def flagFunction(method, name=None):
    """
    Determine whether a function is an optional handler for a I{flag} or an
    I{option}.

    A I{flag} handler takes no additional arguments.  It is used to handle
    command-line arguments like I{--nodaemon}.

    An I{option} handler takes one argument.  It is used to handle command-line
    arguments like I{--path=/foo/bar}.

    @param method: The bound method object to inspect.

    @param name: The name of the option for which the function is a handle.
    @type name: L{str}

    @raise UsageError: If the method takes more than one argument.

    @return: If the method is a flag handler, return C{True}.  Otherwise return
        C{False}.
    """
    reqArgs = len(inspect.signature(method).parameters)
    if reqArgs > 1:
        raise UsageError('Invalid Option function for %s' % (name or method.__name__))
    if reqArgs == 1:
        return False
    return True