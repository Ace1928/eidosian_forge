import functools
import getopt
import pipes
import subprocess
import sys
from humanfriendly import (
from humanfriendly.tables import format_pretty_table, format_smart_table
from humanfriendly.terminal import (
from humanfriendly.terminal.spinners import Spinner
def print_formatted_timespan(value):
    """Print a human readable timespan."""
    output(format_timespan(float(value)))