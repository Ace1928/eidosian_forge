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
def display_version(rev, context):
    if verbose:
        config.print_stdout('Current revision(s) for %s:', util.obfuscate_url_pw(context.connection.engine.url))
    for rev in script.get_all_current(rev):
        config.print_stdout(rev.cmd_format(verbose))
    return []