import logging
import sys
from humanfriendly.compat import interactive_prompt
from humanfriendly.terminal import (
from humanfriendly.text import format, concatenate
def prompt_for_input(question, default=None, padding=True, strip=True):
    """
    Prompt the user for input (free form text).

    :param question: An explanation of what is expected from the user (a string).
    :param default: The return value if the user doesn't enter any text or
                    standard input is not connected to a terminal (which
                    makes it impossible to prompt the user).
    :param padding: Render empty lines before and after the prompt to make it
                    stand out from the surrounding text? (a boolean, defaults
                    to :data:`True`)
    :param strip: Strip leading/trailing whitespace from the user's reply?
    :returns: The text entered by the user (a string) or the value of the
              `default` argument.
    :raises: - :exc:`~exceptions.KeyboardInterrupt` when the program is
               interrupted_ while the prompt is active, for example
               because the user presses Control-C_.
             - :exc:`~exceptions.EOFError` when reading from `standard input`_
               fails, for example because the user presses Control-D_ or
               because the standard input stream is redirected (only if
               `default` is :data:`None`).

    .. _Control-C: https://en.wikipedia.org/wiki/Control-C#In_command-line_environments
    .. _Control-D: https://en.wikipedia.org/wiki/End-of-transmission_character#Meaning_in_Unix
    .. _interrupted: https://en.wikipedia.org/wiki/Unix_signal#SIGINT
    .. _standard input: https://en.wikipedia.org/wiki/Standard_streams#Standard_input_.28stdin.29
    """
    prepare_friendly_prompts()
    reply = None
    try:
        if padding:
            question = '\n' + question
            question = question.replace('\n', '\n ')
        try:
            reply = interactive_prompt(question)
        finally:
            if reply is None:
                sys.stderr.write('\n')
            if padding:
                sys.stderr.write('\n')
    except BaseException as e:
        if isinstance(e, EOFError) and default is not None:
            logger.debug('Got EOF from terminal, returning default value (%r) ..', default)
            return default
        else:
            logger.warning('Interactive prompt was interrupted by exception!', exc_info=True)
            raise
    if default is not None and (not reply):
        return default
    else:
        return reply.strip()