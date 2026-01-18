from yowsup.structs import ProtocolTreeNode
from yowsup.layers.protocol_iq.protocolentities import IqProtocolEntity
def setResultStatusesProps(self, statuses):
    assert type(statuses) is dict, 'statuses must be dict'
    self.statuses = statuses