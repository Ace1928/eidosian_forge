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
def sendGetRequest(self, parser=None, encrypt_params=True, preview=False):
    logger.debug('sendGetRequest(parser=%s, encrypt_params=%s, preview=%s)' % (None if parser is None else '[omitted]', encrypt_params, preview))
    self.response = None
    if encrypt_params:
        logger.debug('Encrypting parameters')
        if logger.level <= logging.DEBUG:
            logger.debug('pre-encrypt (encoded) parameters = \n%s', self.urlencodeParams(self.params))
        params = self.encryptParams(self.params, self.ENC_PUBKEY)
    else:
        params = self.params
    parser = parser or self.parser or ResponseParser()
    headers = dict(list({'User-Agent': self.getUserAgent(), 'Accept': parser.getMeta()}.items()) + list(self.headers.items()))
    host, port, path = self.getConnectionParameters()
    self.response = WARequest.sendRequest(host, port, path, headers, params, 'GET', preview=preview)
    if preview:
        logger.info('Preview request, skip response handling and return None')
        return None
    if not self.response.status == WARequest.OK:
        logger.error('Request not success, status was %s' % self.response.status)
        return {}
    data = self.response.read()
    logger.info(data)
    self.sent = True
    return parser.parse(data.decode(), self.pvars)