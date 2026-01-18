from __future__ import absolute_import
import logging
import socket
import struct
import threading
import time
import serial
from serial.serialutil import SerialBase, SerialException, to_bytes, \
def check_modem_lines(self, force_notification=False):
    """        read control lines from serial port and compare the last value sent to remote.
        send updates on changes.
        """
    modemstate = (self.serial.cts and MODEMSTATE_MASK_CTS) | (self.serial.dsr and MODEMSTATE_MASK_DSR) | (self.serial.ri and MODEMSTATE_MASK_RI) | (self.serial.cd and MODEMSTATE_MASK_CD)
    deltas = modemstate ^ (self.last_modemstate or 0)
    if deltas & MODEMSTATE_MASK_CTS:
        modemstate |= MODEMSTATE_MASK_CTS_CHANGE
    if deltas & MODEMSTATE_MASK_DSR:
        modemstate |= MODEMSTATE_MASK_DSR_CHANGE
    if deltas & MODEMSTATE_MASK_RI:
        modemstate |= MODEMSTATE_MASK_RI_CHANGE
    if deltas & MODEMSTATE_MASK_CD:
        modemstate |= MODEMSTATE_MASK_CD_CHANGE
    if modemstate != self.last_modemstate or force_notification:
        if self._client_is_rfc2217 and modemstate & self.modemstate_mask or force_notification:
            self.rfc2217_send_subnegotiation(SERVER_NOTIFY_MODEMSTATE, to_bytes([modemstate & self.modemstate_mask]))
            if self.logger:
                self.logger.info('NOTIFY_MODEMSTATE: {}'.format(modemstate))
        self.last_modemstate = modemstate & 240