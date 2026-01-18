from midi devices.  It can also list midi devices on the system.
import math
import atexit
import pygame
import pygame.locals
import pygame.pypm as _pypm
def frequency_to_midi(frequency):
    """converts a frequency into a MIDI note.

    Rounds to the closest midi note.

    ::Examples::

    >>> frequency_to_midi(27.5)
    21
    >>> frequency_to_midi(36.7)
    26
    >>> frequency_to_midi(4186.0)
    108
    """
    return int(round(69 + 12 * math.log(frequency / 440.0) / math.log(2)))