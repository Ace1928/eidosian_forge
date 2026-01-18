from yowsup.layers.protocol_messages.protocolentities import MessageProtocolEntity
from yowsup.structs import ProtocolTreeNode
from yowsup.layers.axolotl.protocolentities.enc import EncProtocolEntity
def setEncEntities(self, encEntities=None):
    assert type(encEntities) is list and len(encEntities), 'Must have at least a list of minumum 1 enc entity'
    self.encEntities = encEntities