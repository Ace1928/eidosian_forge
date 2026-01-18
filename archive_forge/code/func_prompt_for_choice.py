import logging
import sys
from humanfriendly.compat import interactive_prompt
from humanfriendly.terminal import (
from humanfriendly.text import format, concatenate
def prompt_for_choice(choices, default=None, padding=True):
    """
    Prompt the user to select a choice from a group of options.

    :param choices: A sequence of strings with available options.
    :param default: The default choice if the user simply presses Enter
                    (expected to be a string, defaults to :data:`None`).
    :param padding: Refer to the documentation of
                    :func:`~humanfriendly.prompts.prompt_for_input()`.
    :returns: The string corresponding to the user's choice.
    :raises: - :exc:`~exceptions.ValueError` if `choices` is an empty sequence.
             - Any exceptions raised by
               :func:`~humanfriendly.prompts.retry_limit()`.
             - Any exceptions raised by
               :func:`~humanfriendly.prompts.prompt_for_input()`.

    When no options are given an exception is raised:

    >>> prompt_for_choice([])
    Traceback (most recent call last):
      File "humanfriendly/prompts.py", line 148, in prompt_for_choice
        raise ValueError("Can't prompt for choice without any options!")
    ValueError: Can't prompt for choice without any options!

    If a single option is given the user isn't prompted:

    >>> prompt_for_choice(['only one choice'])
    'only one choice'

    Here's what the actual prompt looks like by default:

    >>> prompt_for_choice(['first option', 'second option'])
    <BLANKLINE>
      1. first option
      2. second option
    <BLANKLINE>
     Enter your choice as a number or unique substring (Control-C aborts): second
    <BLANKLINE>
    'second option'

    If you don't like the whitespace (empty lines and indentation):

    >>> prompt_for_choice(['first option', 'second option'], padding=False)
     1. first option
     2. second option
    Enter your choice as a number or unique substring (Control-C aborts): first
    'first option'
    """
    indent = ' ' if padding else ''
    choices = list(choices)
    if len(choices) == 1:
        logger.debug("Skipping interactive prompt because there's only option (%r).", choices[0])
        return choices[0]
    elif not choices:
        raise ValueError("Can't prompt for choice without any options!")
    prompt_text = ('\n\n' if padding else '\n').join(['\n'.join([u' %i. %s' % (i, choice) + (' (default choice)' if choice == default else '') for i, choice in enumerate(choices, start=1)]), 'Enter your choice as a number or unique substring (Control-C aborts): '])
    prompt_text = prepare_prompt_text(prompt_text, bold=True)
    logger.debug('Requesting interactive choice on terminal (options are %s) ..', concatenate(map(repr, choices)))
    for attempt in retry_limit():
        reply = prompt_for_input(prompt_text, '', padding=padding, strip=True)
        if not reply and default is not None:
            logger.debug('Default choice selected by empty reply (%r).', default)
            return default
        elif reply.isdigit():
            index = int(reply) - 1
            if 0 <= index < len(choices):
                logger.debug('Option (%r) selected by numeric reply (%s).', choices[index], reply)
                return choices[index]
        matches = []
        for choice in choices:
            lower_reply = reply.lower()
            lower_choice = choice.lower()
            if lower_reply == lower_choice:
                logger.debug('Option (%r) selected by reply (exact match).', choice)
                return choice
            elif lower_reply in lower_choice and len(lower_reply) > 0:
                matches.append(choice)
        if len(matches) == 1:
            logger.debug('Option (%r) selected by reply (substring match on %r).', matches[0], reply)
            return matches[0]
        else:
            if matches:
                details = format("text '%s' matches more than one choice: %s", reply, concatenate(matches))
            elif reply.isdigit():
                details = format('number %i is not a valid choice', int(reply))
            elif reply and (not reply.isspace()):
                details = format("text '%s' doesn't match any choices", reply)
            else:
                details = "there's no default choice"
            logger.debug('Got %s reply (%s), retrying (%i/%i) ..', 'invalid' if reply else 'empty', details, attempt, MAX_ATTEMPTS)
            warning('%sError: Invalid input (%s).', indent, details)