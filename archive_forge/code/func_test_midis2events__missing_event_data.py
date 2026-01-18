import unittest
import pygame
def test_midis2events__missing_event_data(self):
    """Ensures midi events with missing values are handled properly."""
    midi_event_missing_data = ((192, 0, 1), 20000)
    midi_event_missing_timestamp = ((192, 0, 1, 2),)
    for midi_event in (midi_event_missing_data, midi_event_missing_timestamp):
        with self.assertRaises(ValueError):
            events = pygame.midi.midis2events([midi_event], 0)