import functools
import getopt
import pipes
import subprocess
import sys
from humanfriendly import (
from humanfriendly.tables import format_pretty_table, format_smart_table
from humanfriendly.terminal import (
from humanfriendly.terminal.spinners import Spinner
def print_formatted_size(value, binary):
    """Print a human readable size."""
    output(format_size(int(value), binary=binary))