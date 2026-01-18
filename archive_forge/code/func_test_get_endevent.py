import sys
import os
import unittest
import pathlib
import platform
import time
from pygame.tests.test_utils import example_path
import pygame
from pygame import mixer
def test_get_endevent(self):
    """Ensure Channel.get_endevent() returns the correct event type."""
    channel = mixer.Channel(0)
    sound = mixer.Sound(example_path('data/house_lo.wav'))
    channel.play(sound)
    END_EVENT = pygame.USEREVENT + 1
    channel.set_endevent(END_EVENT)
    got_end_event = channel.get_endevent()
    self.assertEqual(got_end_event, END_EVENT)
    channel.stop()
    while channel.get_busy():
        pygame.time.wait(10)
    events = pygame.event.get(got_end_event)
    self.assertTrue(len(events) > 0)