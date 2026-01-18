from io import BytesIO
from dulwich.tests import TestCase
from ..errors import HangupException
from ..protocol import (
def test_mixed(self):
    all_data = b','.join((str(i).encode('ascii') for i in range(100)))
    self.rin.write(all_data)
    self.rin.seek(0)
    data = b''
    for i in range(1, 100):
        data += self.proto.recv(i)
        if len(data) + i > len(all_data):
            data += self.proto.recv(i)
            data += self.proto.recv(1)
            break
        else:
            data += self.proto.read(i)
    else:
        self.fail()
    self.assertEqual(all_data, data)