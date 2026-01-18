import re
from . import utilities
def find_section(text, name):
    """
    Finds all sections of the form

    NAME=BEGINS=HERE
    stuff
    stuff
    stuff
    NAME=ENDS=HERE

    or

    ==NAME=BEGINS==
    stuff
    stuff
    stuff
    ==NAME=ENDS==

    in text where NAME has to be replaced by name.

    >>> t = (
    ... "abc abc\\n"
    ... "==FOO=BEGINS==\\n"
    ... "bar bar\\n"
    ... "==FOO=ENDS==\\n")
    >>> find_section(t, "FOO")
    ['bar bar']
    """
    old_style_regex = name + '=BEGINS=HERE' + '(.*?)' + name + '=ENDS=HERE'
    new_style_regex = '==' + name + '=BEGINS?==' + '(.*?)' + '==' + name + '=ENDS?=='
    regexs = [old_style_regex, new_style_regex]
    if not isinstance(text, str):
        text = text.decode('ascii')
    return [s.strip() for regex in regexs for s in re.findall(regex, text, re.DOTALL)]