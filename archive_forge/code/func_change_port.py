from __future__ import absolute_import
import codecs
import os
import sys
import threading
import serial
from serial.tools.list_ports import comports
from serial.tools import hexlify_codec
def change_port(self):
    """Have a conversation with the user to change the serial port"""
    with self.console:
        try:
            port = ask_for_port()
        except KeyboardInterrupt:
            port = None
    if port and port != self.serial.port:
        self._stop_reader()
        settings = self.serial.getSettingsDict()
        try:
            new_serial = serial.serial_for_url(port, do_not_open=True)
            new_serial.applySettingsDict(settings)
            new_serial.rts = self.serial.rts
            new_serial.dtr = self.serial.dtr
            new_serial.open()
            new_serial.break_condition = self.serial.break_condition
        except Exception as e:
            sys.stderr.write('--- ERROR opening new port: {} ---\n'.format(e))
            new_serial.close()
        else:
            self.serial.close()
            self.serial = new_serial
            sys.stderr.write('--- Port changed to: {} ---\n'.format(self.serial.port))
        self._start_reader()