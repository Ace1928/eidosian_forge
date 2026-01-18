from __future__ import annotations
import collections.abc as c
import contextlib
import json
import os
import re
import shlex
import sys
import tempfile
import textwrap
import typing as t
from .constants import (
from .encoding import (
from .util import (
from .io import (
from .data import (
from .provider.layout import (
from .host_configs import (
def handle_layout_messages(messages: t.Optional[LayoutMessages]) -> None:
    """Display the given layout messages."""
    if not messages:
        return
    for message in messages.info:
        display.info(message, verbosity=1)
    for message in messages.warning:
        display.warning(message)
    if messages.error:
        raise ApplicationError('\n'.join(messages.error))