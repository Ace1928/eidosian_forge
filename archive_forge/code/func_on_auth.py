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
@EventCallback(YowAuthenticationProtocolLayer.EVENT_AUTH)
def on_auth(self, event):
    logger.debug('Received auth event')
    self._profile = self.getProp('profile')
    config = self._profile.config
    local_static = config.client_static_keypair
    username = int(self._profile.username)
    if local_static is None:
        logger.error('client_static_keypair is not defined in specified config, disconnecting')
        self.broadcastEvent(YowLayerEvent(YowNetworkLayer.EVENT_STATE_DISCONNECT, reason='client_static_keypair is not defined in specified config'))
    else:
        if type(local_static) is bytes:
            local_static = KeyPair.from_bytes(local_static)
        assert type(local_static) is KeyPair, type(local_static)
        passive = event.getArg('passive')
        self.setProp(YowNoiseSegmentsLayer.PROP_ENABLED, False)
        if config.edge_routing_info:
            self.toLower(self.EDGE_HEADER)
            self.setProp(YowNoiseSegmentsLayer.PROP_ENABLED, True)
            self.toLower(config.edge_routing_info)
            self.setProp(YowNoiseSegmentsLayer.PROP_ENABLED, False)
        self.toLower(self.HEADER)
        self.setProp(YowNoiseSegmentsLayer.PROP_ENABLED, True)
        remote_static = config.server_static_public
        self._rs = remote_static
        yowsupenv = YowsupEnv.getCurrent()
        client_config = ClientConfig(username=username, passive=passive, useragent=UserAgentConfig(platform=0, app_version=yowsupenv.getVersion(), mcc=config.mcc or '000', mnc=config.mnc or '000', os_version=yowsupenv.getOSVersion(), manufacturer=yowsupenv.getManufacturer(), device=yowsupenv.getDeviceName(), os_build_number=yowsupenv.getOSVersion(), phone_id=config.fdid or '', locale_lang='en', locale_country='US'), pushname=config.pushname or self.DEFAULT_PUSHNAME, short_connect=True)
        if not self._in_handshake():
            logger.debug('Performing handshake [username= %d, passive=%s]' % (username, passive))
            self._handshake_worker = WANoiseProtocolHandshakeWorker(self._wa_noiseprotocol, self._stream, client_config, local_static, remote_static, self.on_handshake_finished)
            logger.debug('Starting handshake worker')
            self._stream.set_events_callback(self._handle_stream_event)
            self._handshake_worker.start()