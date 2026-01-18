import logging
import sys
from humanfriendly.compat import interactive_prompt
from humanfriendly.terminal import (
from humanfriendly.text import format, concatenate
def prepare_prompt_text(prompt_text, **options):
    """
    Wrap a text to be rendered as an interactive prompt in ANSI escape sequences.

    :param prompt_text: The text to render on the prompt (a string).
    :param options: Any keyword arguments are passed on to :func:`.ansi_wrap()`.
    :returns: The resulting prompt text (a string).

    ANSI escape sequences are only used when the standard output stream is
    connected to a terminal. When the standard input stream is connected to a
    terminal any escape sequences are wrapped in "readline hints".
    """
    return ansi_wrap(prompt_text, readline_hints=connected_to_terminal(sys.stdin), **options) if terminal_supports_colors(sys.stdout) else prompt_text