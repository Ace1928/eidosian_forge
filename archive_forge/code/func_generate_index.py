import railroad
from pip._vendor import pyparsing
import typing
from typing import (
from jinja2 import Template
from io import StringIO
import inspect
def generate_index(self) -> int:
    """
        Generate a number used to index a diagram
        """
    self.index += 1
    return self.index