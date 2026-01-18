import os
import re
import sys
from typing import Any, Dict, List, Tuple
import click
import click.parser
import click.shell_completion
from ._completion_shared import (
def get_completion_args(self) -> Tuple[List[str], str]:
    completion_args = os.getenv('_TYPER_COMPLETE_ARGS', '')
    incomplete = os.getenv('_TYPER_COMPLETE_WORD_TO_COMPLETE', '')
    cwords = click.parser.split_arg_string(completion_args)
    args = cwords[1:]
    return (args, incomplete)