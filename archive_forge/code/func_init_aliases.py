import os
import re
import sys
from traitlets.config.configurable import Configurable
from .error import UsageError
from traitlets import List, Instance
from logging import error
import typing as t
def init_aliases(self):
    for name, cmd in self.default_aliases + self.user_aliases:
        if cmd.startswith('ls ') and self.shell is not None and (self.shell.colors == 'NoColor'):
            cmd = cmd.replace(' --color', '')
        self.soft_define_alias(name, cmd)