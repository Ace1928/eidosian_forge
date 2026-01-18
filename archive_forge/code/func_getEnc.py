from yowsup.layers.protocol_messages.protocolentities import MessageProtocolEntity
from yowsup.structs import ProtocolTreeNode
from yowsup.layers.axolotl.protocolentities.enc import EncProtocolEntity
def getEnc(self, encType):
    for enc in self.encEntities:
        if enc.type == encType:
            return enc