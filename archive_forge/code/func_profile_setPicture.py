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
@clicmd('Set profile picture')
def profile_setPicture(self, path):
    if self.assertConnected():
        with PILOptionalModule(failMessage='No PIL library installed, try install pillow') as imp:
            Image = imp('Image')

            def onSuccess(resultIqEntity, originalIqEntity):
                self.output('Profile picture updated successfully')

            def onError(errorIqEntity, originalIqEntity):
                logger.error('Error updating profile picture')
            src = Image.open(path)
            pictureData = src.resize((640, 640)).tobytes('jpeg', 'RGB')
            picturePreview = src.resize((96, 96)).tobytes('jpeg', 'RGB')
            iq = SetPictureIqProtocolEntity(self.getOwnJid(), picturePreview, pictureData)
            self._sendIq(iq, onSuccess, onError)