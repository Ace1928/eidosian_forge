from midi devices.  It can also list midi devices on the system.
import math
import atexit
import pygame
import pygame.locals
import pygame.pypm as _pypm
def pitch_bend(self, value=0, channel=0):
    """modify the pitch of a channel.
        Output.pitch_bend(value=0, channel=0)

        Adjust the pitch of a channel.  The value is a signed integer
        from -8192 to +8191.  For example, 0 means "no change", +4096 is
        typically a semitone higher, and -8192 is 1 whole tone lower (though
        the musical range corresponding to the pitch bend range can also be
        changed in some synthesizers).

        If no value is given, the pitch bend is returned to "no change".
        """
    if not 0 <= channel <= 15:
        raise ValueError('Channel not between 0 and 15.')
    if not -8192 <= value <= 8191:
        raise ValueError(f'Pitch bend value must be between -8192 and +8191, not {value}.')
    value = value + 8192
    lsb = value & 127
    msb = value >> 7
    self.write_short(224 + channel, lsb, msb)