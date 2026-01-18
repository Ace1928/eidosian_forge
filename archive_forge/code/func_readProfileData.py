import os
from .constants import YowConstants
import codecs, sys
import logging
import tempfile
import base64
import hashlib
import os.path, mimetypes
import uuid
from consonance.structs.keypair import KeyPair
from appdirs import user_config_dir
from .optionalmodules import PILOptionalModule, FFVideoOptionalModule
@staticmethod
def readProfileData(profile_name, name, default=None):
    logger.debug('readProfileData(profile_name=%s, name=%s)' % (profile_name, name))
    path = StorageTools.getStorageForProfile(profile_name)
    dataFilePath = os.path.join(path, name)
    if os.path.isfile(dataFilePath):
        logger.debug('Reading %s' % dataFilePath)
        with open(dataFilePath, 'rb') as attrFile:
            return attrFile.read()
    else:
        logger.debug('%s does not exist' % dataFilePath)
    return default