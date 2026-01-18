import re
import time
import platform
from collections import OrderedDict
import six
def resolve_sequence(text, mapper, codes):
    """
    Return a single :class:`Keystroke` instance for given sequence ``text``.

    :arg str text: string of characters received from terminal input stream.
    :arg OrderedDict mapper: unicode multibyte sequences, such as ``u'\\x1b[D'``
        paired by their integer value (260)
    :arg dict codes: a :type:`dict` of integer values (such as 260) paired
        by their mnemonic name, such as ``'KEY_LEFT'``.
    :rtype: Keystroke
    :returns: Keystroke instance for the given sequence

    The given ``text`` may extend beyond a matching sequence, such as
    ``u\\x1b[Dxxx`` returns a :class:`Keystroke` instance of attribute
    :attr:`Keystroke.sequence` valued only ``u\\x1b[D``.  It is up to
    calls to determine that ``xxx`` remains unresolved.
    """
    for sequence, code in mapper.items():
        if text.startswith(sequence):
            return Keystroke(ucs=sequence, code=code, name=codes[code])
    return Keystroke(ucs=text and text[0] or u'')