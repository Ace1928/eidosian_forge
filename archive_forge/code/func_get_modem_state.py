from __future__ import absolute_import
import logging
import socket
import struct
import threading
import time
import serial
from serial.serialutil import SerialBase, SerialException, to_bytes, \
def get_modem_state(self):
    """        get last modem state (cached value. If value is "old", request a new
        one. This cache helps that we don't issue to many requests when e.g. all
        status lines, one after the other is queried by the user (CTS, DSR
        etc.)
        """
    if self._poll_modem_state and self._modemstate_timeout.expired():
        if self.logger:
            self.logger.debug('polling modem state')
        self.rfc2217_send_subnegotiation(NOTIFY_MODEMSTATE)
        timeout = Timeout(self._network_timeout)
        while not timeout.expired():
            time.sleep(0.05)
            if not self._modemstate_timeout.expired():
                break
        else:
            if self.logger:
                self.logger.warning('poll for modem state failed')
    if self._modemstate is not None:
        if self.logger:
            self.logger.debug('using cached modem state')
        return self._modemstate
    else:
        raise SerialException('remote sends no NOTIFY_MODEMSTATE')