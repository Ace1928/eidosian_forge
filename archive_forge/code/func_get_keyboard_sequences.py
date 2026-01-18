import re
import time
import platform
from collections import OrderedDict
import six
def get_keyboard_sequences(term):
    """
    Return mapping of keyboard sequences paired by keycodes.

    :arg blessed.Terminal term: :class:`~.Terminal` instance.
    :returns: mapping of keyboard unicode sequences paired by keycodes
        as integer.  This is used as the argument ``mapper`` to
        the supporting function :func:`resolve_sequence`.
    :rtype: OrderedDict

    Initialize and return a keyboard map and sequence lookup table,
    (sequence, keycode) from :class:`~.Terminal` instance ``term``,
    where ``sequence`` is a multibyte input sequence of unicode
    characters, such as ``u'\\x1b[D'``, and ``keycode`` is an integer
    value, matching curses constant such as term.KEY_LEFT.

    The return value is an OrderedDict instance, with their keys
    sorted longest-first.
    """
    sequence_map = dict(((seq.decode('latin1'), val) for seq, val in ((curses.tigetstr(cap), val) for val, cap in capability_names.items()) if seq) if term.does_styling else ())
    sequence_map.update(_alternative_left_right(term))
    sequence_map.update(DEFAULT_SEQUENCE_MIXIN)
    return OrderedDict(((seq, sequence_map[seq]) for seq in sorted(sequence_map.keys(), key=len, reverse=True)))