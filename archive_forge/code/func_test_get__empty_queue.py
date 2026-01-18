import collections
import time
import unittest
import os
import pygame
def test_get__empty_queue(self):
    """Ensure get() works correctly on an empty queue."""
    expected_events = []
    pygame.event.clear()
    retrieved_events = pygame.event.get()
    self.assertListEqual(retrieved_events, expected_events)
    for event_type in EVENT_TYPES:
        retrieved_events = pygame.event.get(event_type)
        self.assertListEqual(retrieved_events, expected_events)
    retrieved_events = pygame.event.get(EVENT_TYPES)
    self.assertListEqual(retrieved_events, expected_events)