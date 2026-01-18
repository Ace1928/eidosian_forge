import unittest
import pygame
def test_Output(self):
    i = pygame.midi.get_default_output_id()
    if self.midi_output:
        self.assertEqual(self.midi_output.device_id, i)
    i = pygame.midi.get_default_input_id()
    self.assertRaises(pygame.midi.MidiException, pygame.midi.Output, i)
    self.assertRaises(pygame.midi.MidiException, pygame.midi.Output, 9009)
    self.assertRaises(pygame.midi.MidiException, pygame.midi.Output, -1)
    self.assertRaises(TypeError, pygame.midi.Output, '1234')
    self.assertRaises(OverflowError, pygame.midi.Output, pow(2, 99))