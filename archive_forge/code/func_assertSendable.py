import io
from oslotest import base
from oslo_privsep import comm
def assertSendable(self, value):
    self.assertEqual(value, self.send(value))