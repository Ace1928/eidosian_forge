import functools
import getopt
import pipes
import subprocess
import sys
from humanfriendly import (
from humanfriendly.tables import format_pretty_table, format_smart_table
from humanfriendly.terminal import (
from humanfriendly.terminal.spinners import Spinner
def print_formatted_length(value):
    """Print a human readable length."""
    if '.' in value:
        output(format_length(float(value)))
    else:
        output(format_length(int(value)))