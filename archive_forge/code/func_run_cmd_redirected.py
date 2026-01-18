import os
import re
import subprocess
import sys
import tempfile
import time
from ray.autoscaler._private.cli_logger import cf, cli_logger
def run_cmd_redirected(cmd, process_runner=subprocess, silent=False, use_login_shells=False):
    """Run a command and optionally redirect output to a file.

    Args:
        cmd (List[str]): Command to run.
        process_runner: Process runner used for executing commands.
        silent: If true, the command output will be silenced completely
                       (redirected to /dev/null), unless verbose logging
                       is enabled. Use this for running utility commands like
                       rsync.
    """
    if silent and cli_logger.verbosity < 1:
        return _run_and_process_output(cmd, process_runner=process_runner, stdout_file=process_runner.DEVNULL, stderr_file=process_runner.DEVNULL, use_login_shells=use_login_shells)
    if not is_output_redirected():
        return _run_and_process_output(cmd, process_runner=process_runner, stdout_file=sys.stdout, stderr_file=sys.stderr, use_login_shells=use_login_shells)
    else:
        tmpfile_path = os.path.join(tempfile.gettempdir(), 'ray-up-{}-{}.txt'.format(cmd[0], time.time()))
        with open(tmpfile_path, mode='w', buffering=1) as tmp:
            cli_logger.verbose('Command stdout is redirected to {}', cf.bold(tmp.name))
            return _run_and_process_output(cmd, process_runner=process_runner, stdout_file=tmp, stderr_file=tmp, use_login_shells=use_login_shells)