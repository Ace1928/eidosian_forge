from yowsup.layers import YowLayer, YowLayerEvent, YowProtocolLayer
from .protocolentities import *
import logging
def recvIb(self, node):
    if node.getChild('dirty'):
        self.toUpper(DirtyIbProtocolEntity.fromProtocolTreeNode(node))
    elif node.getChild('offline'):
        self.toUpper(OfflineIbProtocolEntity.fromProtocolTreeNode(node))
    elif node.getChild('account'):
        self.toUpper(AccountIbProtocolEntity.fromProtocolTreeNode(node))
    elif node.getChild('edge_routing'):
        logger.debug('ignoring edge_routing ib node for now')
    elif node.getChild('attestation'):
        logger.debug('ignoring attestation ib node for now')
    elif node.getChild('fbip'):
        logger.debug('ignoring fbip ib node for now')
    else:
        logger.warning('Unsupported ib node: %s' % node)