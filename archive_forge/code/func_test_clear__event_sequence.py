import collections
import time
import unittest
import os
import pygame
def test_clear__event_sequence(self):
    """Ensure a sequence of event types can be cleared from the queue."""
    cleared_event_types = EVENT_TYPES[:5]
    expected_event_types = EVENT_TYPES[5:10]
    expected_events = []
    for etype in cleared_event_types:
        pygame.event.post(pygame.event.Event(etype, **EVENT_TEST_PARAMS[etype]))
    for etype in expected_events:
        expected_events.append(pygame.event.Event(etype, **EVENT_TEST_PARAMS[etype]))
        pygame.event.post(expected_events[-1])
    pygame.event.clear(cleared_event_types)
    remaining_events = pygame.event.get()
    self._assertCountEqual(remaining_events, expected_events)