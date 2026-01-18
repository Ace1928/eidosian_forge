import struct
from oslo_log import log as logging
@property
def format_match(self):
    if not self.region('header').complete:
        return False
    signature, = struct.unpack('<I', self.region('header').data[64:68])
    return signature == 3201962111