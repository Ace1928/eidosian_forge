import os
import signal
import threading
import weakref
from breezy import tests, transport
from breezy.bzr.smart import client, medium, server, signals
def test_keyboard_interrupt_propagated(self):

    def call_me_and_raise():
        raise KeyboardInterrupt()
    signals.register_on_hangup('myid', call_me_and_raise)
    self.assertRaises(KeyboardInterrupt, signals._sighup_handler, SIGHUP, None)
    signals.unregister_on_hangup('myid')