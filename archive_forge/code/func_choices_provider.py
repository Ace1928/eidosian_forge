from the CompletionItem.description value.
import argparse
import re
import sys
from argparse import (
from gettext import (
from typing import (
from . import (
@property
def choices_provider(self) -> ChoicesProviderFunc:
    if not isinstance(self.to_call, (ChoicesProviderFuncBase, ChoicesProviderFuncWithTokens)):
        raise ValueError('Function is not a ChoicesProviderFunc')
    return self.to_call