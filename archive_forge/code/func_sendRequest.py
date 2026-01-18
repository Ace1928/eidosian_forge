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
@classmethod
def sendRequest(cls, host, port, path, headers, params, reqType='GET', preview=False):
    logger.debug('sendRequest(host=%s, port=%s, path=%s, headers=%s, params=%s, reqType=%s, preview=%s)' % (host, port, path, headers, params, reqType, preview))
    params = cls.urlencodeParams(params)
    path = path + '?' + params if reqType == 'GET' and params else path
    if not preview:
        logger.debug('Opening connection to %s' % host)
        conn = httplib.HTTPSConnection(host, port) if port == 443 else httplib.HTTPConnection(host, port)
    else:
        logger.debug('Should open connection to %s, but this is a preview' % host)
        conn = None
    if not preview:
        logger.debug('Sending %s request to %s' % (reqType, path))
        conn.request(reqType, path, params, headers)
    else:
        logger.debug('Should send %s request to %s, but this is a preview' % (reqType, path))
        return None
    response = conn.getresponse()
    return response