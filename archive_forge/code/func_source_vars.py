import os
import re
import sys
from typing import Any, Dict, List, Tuple
import click
import click.parser
import click.shell_completion
from ._completion_shared import (
def source_vars(self) -> Dict[str, Any]:
    return {'complete_func': self.func_name, 'autocomplete_var': self.complete_var, 'prog_name': self.prog_name}