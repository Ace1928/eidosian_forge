import os
import sys
import threading
import time
import unittest.mock
import pytest
from cherrypy.process import wspbus
def test_custom_channels(bus, listener):
    """Test that custom pub-sub channels work as built-in ones."""
    expected = []
    custom_listeners = ('hugh', 'louis', 'dewey')
    for channel in custom_listeners:
        for index, priority in enumerate([None, 10, 60, 40]):
            bus.subscribe(channel, listener.get_listener(channel, index), priority)
    for channel in custom_listeners:
        bus.publish(channel, 'ah so')
        expected.extend((msg % (i, channel, 'ah so') for i in (1, 3, 0, 2)))
        bus.publish(channel)
        expected.extend((msg % (i, channel, None) for i in (1, 3, 0, 2)))
    assert listener.responses == expected