import numbers
import random
import re
import string
import textwrap
def generate_slug(text, delimiter='-'):
    """
    Convert text to a normalized "slug" without whitespace.

    :param text: The original text, for example ``Some Random Text!``.
    :param delimiter: The delimiter used to separate words
                      (defaults to the ``-`` character).
    :returns: The slug text, for example ``some-random-text``.
    :raises: :exc:`~exceptions.ValueError` when the provided
             text is nonempty but results in an empty slug.
    """
    slug = text.lower()
    escaped = delimiter.replace('\\', '\\\\')
    slug = re.sub('[^a-z0-9]+', escaped, slug)
    slug = slug.strip(delimiter)
    if text and (not slug):
        msg = 'The provided text %r results in an empty slug!'
        raise ValueError(format(msg, text))
    return slug