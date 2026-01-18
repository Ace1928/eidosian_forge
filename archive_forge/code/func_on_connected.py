from yowsup.layers.interface import YowInterfaceLayer, ProtocolEntityCallback
from yowsup.layers.protocol_media.protocolentities import *
from yowsup.layers.network.layer import YowNetworkLayer
from yowsup.layers import EventCallback
import sys
from ..common.sink_worker import SinkWorker
import tempfile
import logging
import os
@EventCallback(YowNetworkLayer.EVENT_STATE_CONNECTED)
def on_connected(self, event):
    logger.info('Connected, starting SinkWorker')
    storage_dir = self.getProp(self.PROP_STORAGE_DIR)
    if storage_dir is None:
        logger.debug('No storage dir specified, creating tempdir')
        storage_dir = tempfile.mkdtemp('yowsup_mediasink')
    if not os.path.exists(storage_dir):
        logger.debug('%s does not exist, creating' % storage_dir)
        os.makedirs(storage_dir)
    logger.info('Storing incoming media to %s' % storage_dir)
    self._sink_worker = SinkWorker(storage_dir)
    self._sink_worker.start()