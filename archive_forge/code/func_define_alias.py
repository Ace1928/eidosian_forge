import os
import re
import sys
from traitlets.config.configurable import Configurable
from .error import UsageError
from traitlets import List, Instance
from logging import error
import typing as t
def define_alias(self, name, cmd):
    """Define a new alias after validating it.

        This will raise an :exc:`AliasError` if there are validation
        problems.
        """
    caller = Alias(shell=self.shell, name=name, cmd=cmd)
    self.shell.magics_manager.register_function(caller, magic_kind='line', magic_name=name)