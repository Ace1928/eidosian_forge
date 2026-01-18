import unittest
import pygame
def test_midis2events(self):
    """Ensures midi events are properly converted to pygame events."""
    MIDI_DATA = 0
    MD_STATUS = 0
    MD_DATA1 = 1
    MD_DATA2 = 2
    MD_DATA3 = 3
    TIMESTAMP = 1
    midi_events = (((192, 0, 1, 2), 20000), ((144, 60, 1000, 'string_data'), 20001), (('0', '1', '2', '3'), '4'))
    expected_num_events = len(midi_events)
    for device_id in range(3):
        pg_events = pygame.midi.midis2events(midi_events, device_id)
        self.assertEqual(len(pg_events), expected_num_events)
        for i, pg_event in enumerate(pg_events):
            midi_event = midi_events[i]
            midi_event_data = midi_event[MIDI_DATA]
            self.assertEqual(pg_event.__class__.__name__, 'Event')
            self.assertEqual(pg_event.type, pygame.MIDIIN)
            self.assertEqual(pg_event.status, midi_event_data[MD_STATUS])
            self.assertEqual(pg_event.data1, midi_event_data[MD_DATA1])
            self.assertEqual(pg_event.data2, midi_event_data[MD_DATA2])
            self.assertEqual(pg_event.data3, midi_event_data[MD_DATA3])
            self.assertEqual(pg_event.timestamp, midi_event[TIMESTAMP])
            self.assertEqual(pg_event.vice_id, device_id)