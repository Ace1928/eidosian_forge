from __future__ import annotations
import argparse
import enum
import functools
import logging
from typing import Any
from typing import Callable
from typing import Sequence
from flake8 import utils
from flake8.plugins.finder import Plugins
@property
def filtered_option_kwargs(self) -> dict[str, Any]:
    """Return any actually-specified arguments."""
    return {k: v for k, v in self.option_kwargs.items() if v is not _ARG.NO}