import argparse
from typing import (
from . import (
from .argparse_custom import (
from .command_definition import (
from .exceptions import (
from .parsing import (
from .utils import (
def cat_decorator(func: CommandFunc) -> CommandFunc:
    from .utils import categorize
    categorize(func, category)
    return func