from midi devices.  It can also list midi devices on the system.
import math
import atexit
import pygame
import pygame.locals
import pygame.pypm as _pypm
def note_off(self, note, velocity=0, channel=0):
    """turns a midi note off.  Note must be on.
        Output.note_off(note, velocity=0, channel=0)

        note is an integer from 0 to 127
        velocity is an integer from 0 to 127 (release velocity)
        channel is an integer from 0 to 15

        Turn a note off in the output stream.  The note must already
        be on for this to work correctly.
        """
    if not 0 <= channel <= 15:
        raise ValueError('Channel not between 0 and 15.')
    self.write_short(128 + channel, note, velocity)