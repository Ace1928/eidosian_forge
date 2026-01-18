import csv
import functools
import logging
import re
from importlib import import_module
from humanfriendly.compat import StringIO
from humanfriendly.text import dedent, split_paragraphs, trim_empty_lines
def find_meta_variables(usage_text):
    """
    Find the meta variables in the given usage message.

    :param usage_text: The usage message to parse (a string).
    :returns: A list of strings with any meta variables found in the usage
              message.

    When a command line option requires an argument, the convention is to
    format such options as ``--option=ARG``. The text ``ARG`` in this example
    is the meta variable.
    """
    meta_variables = set()
    for match in USAGE_PATTERN.finditer(usage_text):
        token = match.group(0)
        if token.startswith('-'):
            option, _, value = token.partition('=')
            if value:
                meta_variables.add(value)
    return list(meta_variables)