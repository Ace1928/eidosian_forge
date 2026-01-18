import collections
import io
import random as random_module
import string
def named_char_class(char_class, min_count=0):
    """Return a predefined character class.

    The result of this function can be passed to :func:`generate_password` as
    one of the character classes to use in generating a password.

    :param char_class: Any of the character classes named in
                       :const:`CHARACTER_CLASSES`
    :param min_count: The minimum number of members of this class to appear in
                      a generated password
    """
    assert char_class in CHARACTER_CLASSES
    return CharClass(frozenset(_char_class_members[char_class]), min_count)