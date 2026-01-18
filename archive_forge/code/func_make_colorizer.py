import logging
import sys
from typing import Union
from rq.defaults import DEFAULT_LOGGING_DATE_FORMAT, DEFAULT_LOGGING_FORMAT
def make_colorizer(color: str):
    """Creates a function that colorizes text with the given color.

    For example::

        ..codeblock::python

            >>> green = make_colorizer('darkgreen')
            >>> red = make_colorizer('red')
            >>>
            >>> # You can then use:
            >>> print("It's either " + green('OK') + ' or ' + red('Oops'))
    """

    def inner(text):
        return colorizer.colorize(color, text)
    return inner