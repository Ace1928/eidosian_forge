from os import environ, path
from sys import platform as _sys_platform
from re import match, split, search, MULTILINE, IGNORECASE
from kivy.compat import string_types
def escape_markup(text):
    """
    Escape markup characters found in the text. Intended to be used when markup
    text is activated on the Label::

        untrusted_text = escape_markup('Look at the example [1]')
        text = '[color=ff0000]' + untrusted_text + '[/color]'
        w = Label(text=text, markup=True)

    .. versionadded:: 1.3.0
    """
    return text.replace('&', '&amp;').replace('[', '&bl;').replace(']', '&br;')