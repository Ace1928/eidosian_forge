import logging
import sys
from humanfriendly.compat import interactive_prompt
from humanfriendly.terminal import (
from humanfriendly.text import format, concatenate
def prompt_for_confirmation(question, default=None, padding=True):
    """
    Prompt the user for confirmation.

    :param question: The text that explains what the user is confirming (a string).
    :param default: The default value (a boolean) or :data:`None`.
    :param padding: Refer to the documentation of :func:`prompt_for_input()`.
    :returns: - If the user enters 'yes' or 'y' then :data:`True` is returned.
              - If the user enters 'no' or 'n' then :data:`False`  is returned.
              - If the user doesn't enter any text or standard input is not
                connected to a terminal (which makes it impossible to prompt
                the user) the value of the keyword argument ``default`` is
                returned (if that value is not :data:`None`).
    :raises: - Any exceptions raised by :func:`retry_limit()`.
             - Any exceptions raised by :func:`prompt_for_input()`.

    When `default` is :data:`False` and the user doesn't enter any text an
    error message is printed and the prompt is repeated:

    >>> prompt_for_confirmation("Are you sure?")
     <BLANKLINE>
     Are you sure? [y/n]
     <BLANKLINE>
     Error: Please enter 'yes' or 'no' (there's no default choice).
     <BLANKLINE>
     Are you sure? [y/n]

    The same thing happens when the user enters text that isn't recognized:

    >>> prompt_for_confirmation("Are you sure?")
     <BLANKLINE>
     Are you sure? [y/n] about what?
     <BLANKLINE>
     Error: Please enter 'yes' or 'no' (the text 'about what?' is not recognized).
     <BLANKLINE>
     Are you sure? [y/n]
    """
    prompt_text = prepare_prompt_text(question, bold=True)
    hint = '[Y/n]' if default else '[y/N]' if default is not None else '[y/n]'
    prompt_text += ' %s ' % prepare_prompt_text(hint, color=HIGHLIGHT_COLOR)
    logger.debug('Requesting interactive confirmation from terminal: %r', ansi_strip(prompt_text).rstrip())
    for attempt in retry_limit():
        reply = prompt_for_input(prompt_text, '', padding=padding, strip=True)
        if reply.lower() in ('y', 'yes'):
            logger.debug('Confirmation granted by reply (%r).', reply)
            return True
        elif reply.lower() in ('n', 'no'):
            logger.debug('Confirmation denied by reply (%r).', reply)
            return False
        elif not reply and default is not None:
            logger.debug('Default choice selected by empty reply (%r).', 'granted' if default else 'denied')
            return default
        else:
            details = "the text '%s' is not recognized" % reply if reply else "there's no default choice"
            logger.debug('Got %s reply (%s), retrying (%i/%i) ..', 'invalid' if reply else 'empty', details, attempt, MAX_ATTEMPTS)
            warning("{indent}Error: Please enter 'yes' or 'no' ({details}).", indent=' ' if padding else '', details=details)