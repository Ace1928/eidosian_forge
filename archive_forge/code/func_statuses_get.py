from .cli import Cli, clicmd
from yowsup.layers.interface import YowInterfaceLayer, ProtocolEntityCallback
from yowsup.layers import YowLayerEvent, EventCallback
from yowsup.layers.network import YowNetworkLayer
import sys
from yowsup.common import YowConstants
import datetime
import time
import os
import logging
import threading
import base64
from yowsup.layers.protocol_groups.protocolentities      import *
from yowsup.layers.protocol_presence.protocolentities    import *
from yowsup.layers.protocol_messages.protocolentities    import *
from yowsup.layers.protocol_ib.protocolentities          import *
from yowsup.layers.protocol_iq.protocolentities          import *
from yowsup.layers.protocol_contacts.protocolentities    import *
from yowsup.layers.protocol_chatstate.protocolentities   import *
from yowsup.layers.protocol_privacy.protocolentities     import *
from yowsup.layers.protocol_media.protocolentities       import *
from yowsup.layers.protocol_media.mediauploader import MediaUploader
from yowsup.layers.protocol_profiles.protocolentities    import *
from yowsup.common.tools import Jid
from yowsup.common.optionalmodules import PILOptionalModule
from yowsup.layers.axolotl.protocolentities.iq_key_get import GetKeysIqProtocolEntity
@clicmd('Request contacts statuses')
def statuses_get(self, contacts):

    def on_success(entity, original_iq_entity):
        status_outs = []
        for jid, status_info in entity.statuses.items():
            status_outs.append('[user=%s status=%s last_updated=%s]' % (jid, status_info[0], status_info[1]))
        self.output('\n'.join(status_outs), tag='statuses_get result')

    def on_error(entity, original_iq):
        logger.error('Failed to get statuses')
    if self.assertConnected():
        entity = GetStatusesIqProtocolEntity([self.aliasToJid(c) for c in contacts.split(',')])
        self._sendIq(entity, on_success, on_error)