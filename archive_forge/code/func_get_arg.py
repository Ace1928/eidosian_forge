import functools
import inspect
import logging
import traceback
import wsme.exc
import wsme.types
from wsme import utils
def get_arg(self, name):
    """
        Returns a :class:`FunctionArgument` from its name
        """
    for arg in self.arguments:
        if arg.name == name:
            return arg
    return None