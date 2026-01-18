import argparse
import inspect
import numbers
from collections import (
from typing import (
from .ansi import (
from .constants import (
from .argparse_custom import (
from .command_definition import (
from .exceptions import (
from .table_creator import (
def set_default_ap_completer_type(completer_type: Type[ArgparseCompleter]) -> None:
    """
    Set the default ArgparseCompleter class for a cmd2 app.

    :param completer_type: Type that is a subclass of ArgparseCompleter.
    """
    global DEFAULT_AP_COMPLETER
    DEFAULT_AP_COMPLETER = completer_type