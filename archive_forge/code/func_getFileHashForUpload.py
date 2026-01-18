from yowsup.common import YowConstants
from yowsup.layers.protocol_iq.protocolentities import IqProtocolEntity
from yowsup.structs import ProtocolTreeNode
import hashlib
import base64
import os
from yowsup.common.tools import WATools
@staticmethod
def getFileHashForUpload(filePath):
    return WATools.getFileHashForUpload(filePath)