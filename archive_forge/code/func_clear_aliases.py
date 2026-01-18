import os
import re
import sys
from traitlets.config.configurable import Configurable
from .error import UsageError
from traitlets import List, Instance
from logging import error
import typing as t
def clear_aliases(self):
    for name, _ in self.aliases:
        self.undefine_alias(name)