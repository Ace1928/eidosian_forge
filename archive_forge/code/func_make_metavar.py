import errno
import inspect
import os
import sys
from enum import Enum
from gettext import gettext as _
from typing import (
import click
import click.core
import click.formatting
import click.parser
import click.shell_completion
import click.types
import click.utils
def make_metavar(self) -> str:
    if self.metavar is not None:
        return self.metavar
    var = (self.name or '').upper()
    if not self.required:
        var = f'[{var}]'
    type_var = self.type.get_metavar(self)
    if type_var:
        var += f':{type_var}'
    if self.nargs != 1:
        var += '...'
    return var