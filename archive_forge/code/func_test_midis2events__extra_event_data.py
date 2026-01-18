import unittest
import pygame
def test_midis2events__extra_event_data(self):
    """Ensures midi events with extra values are handled properly."""
    midi_event_extra_data = ((192, 0, 1, 2, 'extra'), 20000)
    midi_event_extra_timestamp = ((192, 0, 1, 2), 20000, 'extra')
    for midi_event in (midi_event_extra_data, midi_event_extra_timestamp):
        with self.assertRaises(ValueError):
            events = pygame.midi.midis2events([midi_event], 0)