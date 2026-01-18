from __future__ import absolute_import
import codecs
import os
import sys
import threading
import serial
from serial.tools.list_ports import comports
from serial.tools import hexlify_codec
def dump_port_settings(self):
    """Write current settings to sys.stderr"""
    sys.stderr.write('\n--- Settings: {p.name}  {p.baudrate},{p.bytesize},{p.parity},{p.stopbits}\n'.format(p=self.serial))
    sys.stderr.write('--- RTS: {:8}  DTR: {:8}  BREAK: {:8}\n'.format('active' if self.serial.rts else 'inactive', 'active' if self.serial.dtr else 'inactive', 'active' if self.serial.break_condition else 'inactive'))
    try:
        sys.stderr.write('--- CTS: {:8}  DSR: {:8}  RI: {:8}  CD: {:8}\n'.format('active' if self.serial.cts else 'inactive', 'active' if self.serial.dsr else 'inactive', 'active' if self.serial.ri else 'inactive', 'active' if self.serial.cd else 'inactive'))
    except serial.SerialException:
        pass
    sys.stderr.write('--- software flow control: {}\n'.format('active' if self.serial.xonxoff else 'inactive'))
    sys.stderr.write('--- hardware flow control: {}\n'.format('active' if self.serial.rtscts else 'inactive'))
    sys.stderr.write('--- serial input encoding: {}\n'.format(self.input_encoding))
    sys.stderr.write('--- serial output encoding: {}\n'.format(self.output_encoding))
    sys.stderr.write('--- EOL: {}\n'.format(self.eol.upper()))
    sys.stderr.write('--- filters: {}\n'.format(' '.join(self.filters)))