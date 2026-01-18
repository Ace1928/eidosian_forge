from __future__ import annotations
import os
from typing import List
from typing import Optional
from typing import TYPE_CHECKING
from typing import Union
from . import autogenerate as autogen
from . import util
from .runtime.environment import EnvironmentContext
from .script import ScriptDirectory
def do_ensure_version(rev, context):
    context._ensure_version_table()
    return []