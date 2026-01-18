import os
import sys
import threading
import time
import unittest.mock
import pytest
from cherrypy.process import wspbus
def test_graceful(bus, listener, log_tracker):
    """Test that bus graceful state triggers all listeners."""
    num = 3
    for index in range(num):
        bus.subscribe('graceful', listener.get_listener('graceful', index))
    bus.graceful()
    assert set(listener.responses) == set((msg % (i, 'graceful', None) for i in range(num)))
    assert log_tracker.log_entries == ['Bus graceful']