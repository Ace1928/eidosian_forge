import os
import re
import sys
from typing import Any, Dict, List, Tuple
import click
import click.parser
import click.shell_completion
from ._completion_shared import (
def completion_init() -> None:
    click.shell_completion.add_completion_class(BashComplete, Shells.bash.value)
    click.shell_completion.add_completion_class(ZshComplete, Shells.zsh.value)
    click.shell_completion.add_completion_class(FishComplete, Shells.fish.value)
    click.shell_completion.add_completion_class(PowerShellComplete, Shells.powershell.value)
    click.shell_completion.add_completion_class(PowerShellComplete, Shells.pwsh.value)