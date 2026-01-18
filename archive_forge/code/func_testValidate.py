import unittest
import websocket as ws
from websocket._abnf import *
def testValidate(self):
    a_invalid_ping = ABNF(0, 0, 0, 0, opcode=ABNF.OPCODE_PING)
    self.assertRaises(ws._exceptions.WebSocketProtocolException, a_invalid_ping.validate, skip_utf8_validation=False)
    a_bad_rsv_value = ABNF(0, 1, 0, 0, opcode=ABNF.OPCODE_TEXT)
    self.assertRaises(ws._exceptions.WebSocketProtocolException, a_bad_rsv_value.validate, skip_utf8_validation=False)
    a_bad_opcode = ABNF(0, 0, 0, 0, opcode=77)
    self.assertRaises(ws._exceptions.WebSocketProtocolException, a_bad_opcode.validate, skip_utf8_validation=False)
    a_bad_close_frame = ABNF(0, 0, 0, 0, opcode=ABNF.OPCODE_CLOSE, data=b'\x01')
    self.assertRaises(ws._exceptions.WebSocketProtocolException, a_bad_close_frame.validate, skip_utf8_validation=False)
    a_bad_close_frame_2 = ABNF(0, 0, 0, 0, opcode=ABNF.OPCODE_CLOSE, data=b'\x01\x8a\xaa\xff\xdd')
    self.assertRaises(ws._exceptions.WebSocketProtocolException, a_bad_close_frame_2.validate, skip_utf8_validation=False)
    a_bad_close_frame_3 = ABNF(0, 0, 0, 0, opcode=ABNF.OPCODE_CLOSE, data=b'\x03\xe7')
    self.assertRaises(ws._exceptions.WebSocketProtocolException, a_bad_close_frame_3.validate, skip_utf8_validation=True)