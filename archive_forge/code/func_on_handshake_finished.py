from yowsup.layers.noise.workers.handshake import WANoiseProtocolHandshakeWorker
from yowsup.layers import YowLayer, EventCallback
from yowsup.layers.auth.layer_authentication import YowAuthenticationProtocolLayer
from yowsup.layers.network.layer import YowNetworkLayer
from yowsup.layers.noise.layer_noise_segments import YowNoiseSegmentsLayer
from yowsup.config.manager import ConfigManager
from yowsup.env.env import YowsupEnv
from yowsup.layers import YowLayerEvent
from yowsup.structs.protocoltreenode import ProtocolTreeNode
from yowsup.layers.coder.encoder import WriteEncoder
from yowsup.layers.coder.tokendictionary import TokenDictionary
from consonance.protocol import WANoiseProtocol
from consonance.config.client import ClientConfig
from consonance.config.useragent import UserAgentConfig
from consonance.streams.segmented.blockingqueue import BlockingQueueSegmentedStream
from consonance.structs.keypair import KeyPair
import threading
import logging
def on_handshake_finished(self, e=None):
    if e is not None:
        self.emitEvent(YowLayerEvent(self.EVENT_HANDSHAKE_FAILED, reason=e))
        data = WriteEncoder(TokenDictionary()).protocolTreeNodeToBytes(ProtocolTreeNode('failure', {'reason': str(e)}))
        self.toUpper(data)
        logger.error('An error occurred during handshake, try login again.')