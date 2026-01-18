import struct
from os_ken.lib.pack_utils import msg_pack_into
from . import packet_base
from . import packet_utils
from . import ether_types
@flow_id.setter
def flow_id(self, flow_id):
    self._key = self._key & 4294967040 | flow_id
    self._flow_id = flow_id