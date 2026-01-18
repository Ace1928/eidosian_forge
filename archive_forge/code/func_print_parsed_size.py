import functools
import getopt
import pipes
import subprocess
import sys
from humanfriendly import (
from humanfriendly.tables import format_pretty_table, format_smart_table
from humanfriendly.terminal import (
from humanfriendly.terminal.spinners import Spinner
def print_parsed_size(value):
    """Parse a human readable data size and print the number of bytes."""
    output(parse_size(value))