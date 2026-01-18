import numbers
import random
import re
import string
import textwrap
def pluralize_raw(count, singular, plural=None):
    """
    Select the singular or plural form of a word based on a count.

    :param count: The count (a number).
    :param singular: The singular form of the word (a string).
    :param plural: The plural form of the word (a string or :data:`None`).
    :returns: The singular or plural form of the word (a string).

    When the given count is exactly 1.0 the singular form of the word is
    selected, in all other cases the plural form of the word is selected.

    If the plural form of the word is not provided it is obtained by
    concatenating the singular form of the word with the letter "s". Of course
    this will not always be correct, which is why you have the option to
    specify both forms.
    """
    if not plural:
        plural = singular + 's'
    return singular if float(count) == 1.0 else plural