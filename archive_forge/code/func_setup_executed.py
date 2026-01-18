from __future__ import annotations
import abc
import datetime
import os
import re
import tempfile
import time
import typing as t
from ....encoding import (
from ....io import (
from ....util import (
from ....util_common import (
from ....target import (
from ....config import (
from ....ci import (
from ....data import (
from ....docker_util import (
@setup_executed.setter
def setup_executed(self, value: bool) -> None:
    """True if setup has been executed, otherwise False."""
    self._set_cloud_config(self._SETUP_EXECUTED, value)