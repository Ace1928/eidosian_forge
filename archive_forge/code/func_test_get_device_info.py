import unittest
import pygame
def test_get_device_info(self):
    an_id = pygame.midi.get_default_output_id()
    if an_id != -1:
        interf, name, input, output, opened = pygame.midi.get_device_info(an_id)
        self.assertEqual(output, 1)
        self.assertEqual(input, 0)
        self.assertEqual(opened, 0)
    an_in_id = pygame.midi.get_default_input_id()
    if an_in_id != -1:
        r = pygame.midi.get_device_info(an_in_id)
        interf, name, input, output, opened = r
        self.assertEqual(output, 0)
        self.assertEqual(input, 1)
        self.assertEqual(opened, 0)
    out_of_range = pygame.midi.get_count()
    for num in range(out_of_range):
        self.assertIsNotNone(pygame.midi.get_device_info(num))
    info = pygame.midi.get_device_info(out_of_range)
    self.assertIsNone(info)