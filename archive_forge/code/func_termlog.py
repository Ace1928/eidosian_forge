import os
import sys
import tempfile
import time
import click
import wandb
from wandb import env
def termlog(string='', newline=True, repeat=True):
    """Log to standard error with formatting.

    Arguments:
        string (str, optional): The string to print
        newline (bool, optional): Print a newline at the end of the string
        repeat (bool, optional): If set to False only prints the string once per process
    """
    if string:
        line = '\n'.join([f'{LOG_STRING}: {s}' for s in string.split('\n')])
    else:
        line = ''
    if not repeat and line in PRINTED_MESSAGES:
        return
    if len(PRINTED_MESSAGES) < 1000:
        PRINTED_MESSAGES.add(line)
    if os.getenv(env.SILENT):
        from wandb import util
        from wandb.sdk.lib import filesystem
        filesystem.mkdir_exists_ok(os.path.dirname(util.get_log_file_path()))
        with open(util.get_log_file_path(), 'w') as log:
            click.echo(line, file=log, nl=newline)
    else:
        click.echo(line, file=sys.stderr, nl=newline)