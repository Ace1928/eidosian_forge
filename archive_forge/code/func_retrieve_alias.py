import os
import re
import sys
from traitlets.config.configurable import Configurable
from .error import UsageError
from traitlets import List, Instance
from logging import error
import typing as t
def retrieve_alias(self, name):
    """Retrieve the command to which an alias expands."""
    caller = self.get_alias(name)
    if caller:
        return caller.cmd
    else:
        raise ValueError('%s is not an alias' % name)