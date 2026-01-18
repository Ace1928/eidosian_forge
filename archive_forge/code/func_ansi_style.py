import codecs
import numbers
import os
import platform
import re
import subprocess
import sys
from humanfriendly.compat import coerce_string, is_unicode, on_windows, which
from humanfriendly.decorators import cached
from humanfriendly.deprecation import define_aliases
from humanfriendly.text import concatenate, format
from humanfriendly.usage import format_usage
def ansi_style(**kw):
    """
    Generate ANSI escape sequences for the given color and/or style(s).

    :param color: The foreground color. Three types of values are supported:

                  - The name of a color (one of the strings 'black', 'red',
                    'green', 'yellow', 'blue', 'magenta', 'cyan' or 'white').
                  - An integer that refers to the 256 color mode palette.
                  - A tuple or list with three integers representing an RGB
                    (red, green, blue) value.

                  The value :data:`None` (the default) means no escape
                  sequence to switch color will be emitted.
    :param background: The background color (see the description
                       of the `color` argument).
    :param bright: Use high intensity colors instead of default colors
                   (a boolean, defaults to :data:`False`).
    :param readline_hints: If :data:`True` then :func:`readline_wrap()` is
                           applied to the generated ANSI escape sequences (the
                           default is :data:`False`).
    :param kw: Any additional keyword arguments are expected to match a key
               in the :data:`ANSI_TEXT_STYLES` dictionary. If the argument's
               value evaluates to :data:`True` the respective style will be
               enabled.
    :returns: The ANSI escape sequences to enable the requested text styles or
              an empty string if no styles were requested.
    :raises: :exc:`~exceptions.ValueError` when an invalid color name is given.

    Even though only eight named colors are supported, the use of `bright=True`
    and `faint=True` increases the number of available colors to around 24 (it
    may be slightly lower, for example because faint black is just black).

    **Support for 8-bit colors**

    In `release 4.7`_ support for 256 color mode was added. While this
    significantly increases the available colors it's not very human friendly
    in usage because you need to look up color codes in the `256 color mode
    palette <https://en.wikipedia.org/wiki/ANSI_escape_code#8-bit>`_.

    You can use the ``humanfriendly --demo`` command to get a demonstration of
    the available colors, see also the screen shot below. Note that the small
    font size in the screen shot was so that the demonstration of 256 color
    mode support would fit into a single screen shot without scrolling :-)
    (I wasn't feeling very creative).

      .. image:: images/ansi-demo.png

    **Support for 24-bit colors**

    In `release 4.14`_ support for 24-bit colors was added by accepting a tuple
    or list with three integers representing the RGB (red, green, blue) value
    of a color. This is not included in the demo because rendering millions of
    colors was deemed unpractical ;-).

    .. _release 4.7: http://humanfriendly.readthedocs.io/en/latest/changelog.html#release-4-7-2018-01-14
    .. _release 4.14: http://humanfriendly.readthedocs.io/en/latest/changelog.html#release-4-14-2018-07-13
    """
    sequences = [ANSI_TEXT_STYLES[k] for k, v in kw.items() if k in ANSI_TEXT_STYLES and v]
    for color_type in ('color', 'background'):
        color_value = kw.get(color_type)
        if isinstance(color_value, (tuple, list)):
            if len(color_value) != 3:
                msg = 'Invalid color value %r! (expected tuple or list with three numbers)'
                raise ValueError(msg % color_value)
            sequences.append(48 if color_type == 'background' else 38)
            sequences.append(2)
            sequences.extend(map(int, color_value))
        elif isinstance(color_value, numbers.Number):
            sequences.extend((39 if color_type == 'background' else 38, 5, int(color_value)))
        elif color_value:
            if color_value not in ANSI_COLOR_CODES:
                msg = 'Invalid color value %r! (expected an integer or one of the strings %s)'
                raise ValueError(msg % (color_value, concatenate(map(repr, sorted(ANSI_COLOR_CODES)))))
            offset = (100 if kw.get('bright') else 40) if color_type == 'background' else 90 if kw.get('bright') else 30
            sequences.append(offset + ANSI_COLOR_CODES[color_value])
    if sequences:
        encoded = ANSI_CSI + ';'.join(map(str, sequences)) + ANSI_SGR
        return readline_wrap(encoded) if kw.get('readline_hints') else encoded
    else:
        return ''