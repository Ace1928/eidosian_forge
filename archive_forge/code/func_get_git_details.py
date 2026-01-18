from __future__ import annotations
import abc
import base64
import json
import os
import tempfile
import typing as t
from ..encoding import (
from ..io import (
from ..config import (
from ..util import (
@abc.abstractmethod
def get_git_details(self, args: CommonConfig) -> t.Optional[dict[str, t.Any]]:
    """Return details about git in the current environment."""