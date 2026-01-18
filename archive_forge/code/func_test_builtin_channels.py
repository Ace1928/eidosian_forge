import os
import sys
import threading
import time
import unittest.mock
import pytest
from cherrypy.process import wspbus
def test_builtin_channels(bus, listener):
    """Test that built-in channels trigger corresponding listeners."""
    expected = []
    for channel in bus.listeners:
        for index, priority in enumerate([100, 50, 0, 51]):
            bus.subscribe(channel, listener.get_listener(channel, index), priority)
    for channel in bus.listeners:
        bus.publish(channel)
        expected.extend([msg % (i, channel, None) for i in (2, 1, 3, 0)])
        bus.publish(channel, arg=79347)
        expected.extend([msg % (i, channel, 79347) for i in (2, 1, 3, 0)])
    assert listener.responses == expected