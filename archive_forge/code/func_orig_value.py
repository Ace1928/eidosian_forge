from the CompletionItem.description value.
import argparse
import re
import sys
from argparse import (
from gettext import (
from typing import (
from . import (
@property
def orig_value(self) -> Any:
    """Read-only property for _orig_value"""
    return self._orig_value