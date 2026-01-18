import unittest
import pygame
def test_midiout(self):
    """Ensures the MIDIOUT event id exists in the midi module.

        The MIDIOUT event id can be accessed via the midi module for backward
        compatibility.
        """
    self.assertEqual(pygame.midi.MIDIOUT, pygame.MIDIOUT)
    self.assertEqual(pygame.midi.MIDIOUT, pygame.locals.MIDIOUT)
    self.assertNotEqual(pygame.midi.MIDIOUT, pygame.MIDIIN)
    self.assertNotEqual(pygame.midi.MIDIOUT, pygame.locals.MIDIIN)