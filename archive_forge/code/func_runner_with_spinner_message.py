import logging
import os
import shlex
import subprocess
from typing import (
from pip._vendor.rich.markup import escape
from pip._internal.cli.spinners import SpinnerInterface, open_spinner
from pip._internal.exceptions import InstallationSubprocessError
from pip._internal.utils.logging import VERBOSE, subprocess_logger
from pip._internal.utils.misc import HiddenText
def runner_with_spinner_message(message: str) -> Callable[..., None]:
    """Provide a subprocess_runner that shows a spinner message.

    Intended for use with for BuildBackendHookCaller. Thus, the runner has
    an API that matches what's expected by BuildBackendHookCaller.subprocess_runner.
    """

    def runner(cmd: List[str], cwd: Optional[str]=None, extra_environ: Optional[Mapping[str, Any]]=None) -> None:
        with open_spinner(message) as spinner:
            call_subprocess(cmd, command_desc=message, cwd=cwd, extra_environ=extra_environ, spinner=spinner)
    return runner