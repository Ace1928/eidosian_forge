from yowsup.layers import YowLayer, YowLayerEvent, YowProtocolLayer
from .protocolentities import *
from yowsup.layers.protocol_iq.protocolentities import ErrorIqProtocolEntity
def sendPresence(self, entity):
    self.entityToLower(entity)