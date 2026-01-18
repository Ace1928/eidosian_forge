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
def encryptParams(self, params, key):
    """
        :param params:
        :type params: list
        :param key:
        :type key: ECPublicKey
        :return:
        :rtype: list
        """
    keypair = Curve.generateKeyPair()
    encodedparams = self.urlencodeParams(params)
    cipher = AESGCM(Curve.calculateAgreement(key, keypair.privateKey))
    ciphertext = cipher.encrypt(b'\x00\x00\x00\x00' + struct.pack('>Q', 0), encodedparams.encode(), b'')
    payload = base64.b64encode(keypair.publicKey.serialize()[1:] + ciphertext)
    return [('ENC', payload)]