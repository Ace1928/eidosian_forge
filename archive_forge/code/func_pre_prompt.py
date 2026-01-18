from typing import Any, Generic, List, Optional, TextIO, TypeVar, Union, overload
from . import get_console
from .console import Console
from .text import Text, TextType
def pre_prompt(self) -> None:
    """Hook to display something before the prompt."""