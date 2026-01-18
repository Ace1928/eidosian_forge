import collections
import time
import unittest
import os
import pygame
def test_event_name__userevent_boundary(self):
    """Ensures event_name() does not return 'UserEvent' for events
        just outside the user event range.
        """
    unexpected_name = 'UserEvent'
    for event in (pygame.USEREVENT - 1, pygame.NUMEVENTS):
        self.assertNotEqual(pygame.event.event_name(event), unexpected_name, f'0x{event:X}')