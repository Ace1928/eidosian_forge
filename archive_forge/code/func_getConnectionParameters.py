from .waresponseparser import ResponseParser
from yowsup.env import YowsupEnv
import sys
import logging
from axolotl.ecc.curve import Curve
from axolotl.ecc.ec import ECPublicKey
from yowsup.common.tools import WATools
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from yowsup.config.v1.config import Config
from yowsup.profile.profile import YowProfile
import struct
import random
import base64
def getConnectionParameters(self):
    if not self.url:
        return ('', '', self.port)
    try:
        url = self.url.split('://', 1)
        url = url[0] if len(url) == 1 else url[1]
        host, path = url.split('/', 1)
    except ValueError:
        host = url
        path = ''
    path = '/' + path
    return (host, self.port, path)