from the CompletionItem.description value.
import argparse
import re
import sys
from argparse import (
from gettext import (
from typing import (
from . import (
def set_default_argument_parser_type(parser_type: Type[argparse.ArgumentParser]) -> None:
    """
    Set the default ArgumentParser class for a cmd2 app. This must be called prior to loading cmd2.py if
    you want to override the parser for cmd2's built-in commands. See examples/override_parser.py.
    """
    global DEFAULT_ARGUMENT_PARSER
    DEFAULT_ARGUMENT_PARSER = parser_type