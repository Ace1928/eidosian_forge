from midi devices.  It can also list midi devices on the system.
import math
import atexit
import pygame
import pygame.locals
import pygame.pypm as _pypm
def midi_to_ansi_note(midi_note):
    """returns the Ansi Note name for a midi number.

    ::Examples::

    >>> midi_to_ansi_note(21)
    'A0'
    >>> midi_to_ansi_note(102)
    'F#7'
    >>> midi_to_ansi_note(108)
    'C8'
    """
    notes = ['A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#']
    num_notes = 12
    note_name = notes[int((midi_note - 21) % num_notes)]
    note_number = (midi_note - 12) // num_notes
    return f'{note_name}{note_number}'