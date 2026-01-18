import unittest
import pygame
def test_midiin(self):
    """Ensures the MIDIIN event id exists in the midi module.

        The MIDIIN event id can be accessed via the midi module for backward
        compatibility.
        """
    self.assertEqual(pygame.midi.MIDIIN, pygame.MIDIIN)
    self.assertEqual(pygame.midi.MIDIIN, pygame.locals.MIDIIN)
    self.assertNotEqual(pygame.midi.MIDIIN, pygame.MIDIOUT)
    self.assertNotEqual(pygame.midi.MIDIIN, pygame.locals.MIDIOUT)