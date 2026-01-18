from yowsup.common import YowConstants
from yowsup.layers.protocol_iq.protocolentities import IqProtocolEntity
from yowsup.structs import ProtocolTreeNode
@jids.setter
def jids(self, value):
    assert type(value) is list, 'expected list of jids, got %s' % type(value)
    self._jids = value