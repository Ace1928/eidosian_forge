import functools
import getopt
import logging
import sys
import tempfile
import webbrowser
from humanfriendly.terminal import connected_to_terminal, output, usage, warning
from coloredlogs.converter import capture, convert
from coloredlogs.demo import demonstrate_colored_logging
def convert_command_output(*command):
    """
    Command line interface for ``coloredlogs --to-html``.

    Takes a command (and its arguments) and runs the program under ``script``
    (emulating an interactive terminal), intercepts the output of the command
    and converts ANSI escape sequences in the output to HTML.
    """
    captured_output = capture(command)
    converted_output = convert(captured_output)
    if connected_to_terminal():
        fd, temporary_file = tempfile.mkstemp(suffix='.html')
        with open(temporary_file, 'w') as handle:
            handle.write(converted_output)
        webbrowser.open(temporary_file)
    elif captured_output and (not captured_output.isspace()):
        output(converted_output)