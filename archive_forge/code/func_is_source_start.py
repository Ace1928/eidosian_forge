import sys
import re
from abc import ABCMeta
from unicodedata import name as unicode_name
from decimal import Decimal, DecimalException
from typing import Any, cast, overload, Callable, Dict, Generic, List, \
def is_source_start(self) -> bool:
    """
        Returns `True` if the parser is positioned at the start
        of the source, ignoring the spaces.
        """
    return self.token.is_source_start()