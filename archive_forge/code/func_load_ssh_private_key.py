from __future__ import annotations
import binascii
import enum
import os
import re
import typing
import warnings
from base64 import encodebytes as _base64_encode
from dataclasses import dataclass
from cryptography import utils
from cryptography.exceptions import UnsupportedAlgorithm
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import (
from cryptography.hazmat.primitives.asymmetric import utils as asym_utils
from cryptography.hazmat.primitives.ciphers import (
from cryptography.hazmat.primitives.serialization import (
def load_ssh_private_key(data: bytes, password: typing.Optional[bytes], backend: typing.Any=None) -> SSHPrivateKeyTypes:
    """Load private key from OpenSSH custom encoding."""
    utils._check_byteslike('data', data)
    if password is not None:
        utils._check_bytes('password', password)
    m = _PEM_RC.search(data)
    if not m:
        raise ValueError('Not OpenSSH private key format')
    p1 = m.start(1)
    p2 = m.end(1)
    data = binascii.a2b_base64(memoryview(data)[p1:p2])
    if not data.startswith(_SK_MAGIC):
        raise ValueError('Not OpenSSH private key format')
    data = memoryview(data)[len(_SK_MAGIC):]
    ciphername, data = _get_sshstr(data)
    kdfname, data = _get_sshstr(data)
    kdfoptions, data = _get_sshstr(data)
    nkeys, data = _get_u32(data)
    if nkeys != 1:
        raise ValueError('Only one key supported')
    pubdata, data = _get_sshstr(data)
    pub_key_type, pubdata = _get_sshstr(pubdata)
    kformat = _lookup_kformat(pub_key_type)
    pubfields, pubdata = kformat.get_public(pubdata)
    _check_empty(pubdata)
    if (ciphername, kdfname) != (_NONE, _NONE):
        ciphername_bytes = ciphername.tobytes()
        if ciphername_bytes not in _SSH_CIPHERS:
            raise UnsupportedAlgorithm(f'Unsupported cipher: {ciphername_bytes!r}')
        if kdfname != _BCRYPT:
            raise UnsupportedAlgorithm(f'Unsupported KDF: {kdfname!r}')
        blklen = _SSH_CIPHERS[ciphername_bytes].block_len
        tag_len = _SSH_CIPHERS[ciphername_bytes].tag_len
        edata, data = _get_sshstr(data)
        if _SSH_CIPHERS[ciphername_bytes].is_aead:
            tag = bytes(data)
            if len(tag) != tag_len:
                raise ValueError('Corrupt data: invalid tag length for cipher')
        else:
            _check_empty(data)
        _check_block_size(edata, blklen)
        salt, kbuf = _get_sshstr(kdfoptions)
        rounds, kbuf = _get_u32(kbuf)
        _check_empty(kbuf)
        ciph = _init_cipher(ciphername_bytes, password, salt.tobytes(), rounds)
        dec = ciph.decryptor()
        edata = memoryview(dec.update(edata))
        if _SSH_CIPHERS[ciphername_bytes].is_aead:
            assert isinstance(dec, AEADDecryptionContext)
            _check_empty(dec.finalize_with_tag(tag))
        else:
            _check_empty(dec.finalize())
    else:
        edata, data = _get_sshstr(data)
        _check_empty(data)
        blklen = 8
        _check_block_size(edata, blklen)
    ck1, edata = _get_u32(edata)
    ck2, edata = _get_u32(edata)
    if ck1 != ck2:
        raise ValueError('Corrupt data: broken checksum')
    key_type, edata = _get_sshstr(edata)
    if key_type != pub_key_type:
        raise ValueError('Corrupt data: key type mismatch')
    private_key, edata = kformat.load_private(edata, pubfields)
    comment, edata = _get_sshstr(edata)
    if edata != _PADDING[:len(edata)]:
        raise ValueError('Corrupt data: invalid padding')
    if isinstance(private_key, dsa.DSAPrivateKey):
        warnings.warn('SSH DSA keys are deprecated and will be removed in a future release.', utils.DeprecatedIn40, stacklevel=2)
    return private_key