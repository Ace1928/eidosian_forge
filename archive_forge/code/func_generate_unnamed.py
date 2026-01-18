import railroad
from pip._vendor import pyparsing
import typing
from typing import (
from jinja2 import Template
from io import StringIO
import inspect
def generate_unnamed(self) -> int:
    """
        Generate a number used in the name of an otherwise unnamed diagram
        """
    self.unnamed_index += 1
    return self.unnamed_index