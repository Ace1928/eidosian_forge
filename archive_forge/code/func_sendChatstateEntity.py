from yowsup.layers import YowLayer, YowLayerEvent, YowProtocolLayer
from .protocolentities import * 
def sendChatstateEntity(self, entity):
    self.entityToLower(entity)