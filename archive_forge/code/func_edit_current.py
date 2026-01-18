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
def edit_current(rev, context):
    if not rev:
        raise util.CommandError('No current revisions')
    for sc in script.get_revisions(rev):
        util.open_in_editor(sc.path)
    return []