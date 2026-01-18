import argparse
import base64
import json
import os.path
import re
import struct
import sys
import typing
from spnego._asn1 import (
from spnego._context import GSSMech
from spnego._kerberos import (
from spnego._ntlm_raw.crypto import hmac_md5, ntowfv1, ntowfv2, rc4k
from spnego._ntlm_raw.messages import (
from spnego._spnego import InitialContextToken, NegTokenInit, NegTokenResp, unpack_token
from spnego._text import to_bytes
from spnego._tls_struct import (
def parse_tls_token(b_data: bytes) -> typing.List[typing.Dict[str, typing.Any]]:
    """
    :param b_data: A byte string of the TLS token to parse.
    :return: A dict containing the parsed TLS token data.
    """
    view = memoryview(b_data)
    res = []
    while view:
        content_type = TlsContentType(struct.unpack('B', view[:1])[0])
        protocol_version = TlsProtocolVersion(struct.unpack('>H', view[1:3])[0])
        token_length = struct.unpack('>H', view[3:5])[0]
        token_view = view[5:5 + token_length]
        data: typing.Optional[typing.List[typing.Dict[str, typing.Any]]] = None
        if content_type == TlsContentType.handshake:
            data = []
            while token_view:
                handshake_type = TlsHandshakeMessageType(struct.unpack('B', token_view[:1])[0])
                message_len = struct.unpack('>L', b'\x00' + token_view[1:4])[0]
                handshake_view = token_view[4:4 + message_len]
                handshake_data: typing.Optional[typing.Dict[str, typing.Any]] = None
                if handshake_type == TlsHandshakeMessageType.client_hello:
                    handshake_data = _parse_tls_handshake_client_hello(handshake_view)
                elif handshake_type == TlsHandshakeMessageType.server_hello:
                    handshake_data = _parse_tls_handshake_server_hello(handshake_view)
                elif handshake_type == TlsHandshakeMessageType.certificate:
                    cert_len = struct.unpack('>I', b'\x00' + handshake_view[:3])[0]
                    handshake_data = {'Certificate': base64.b16encode(handshake_view[3:3 + cert_len].tobytes()).decode()}
                elif handshake_type == TlsHandshakeMessageType.certificate_request:
                    handshake_data = _parse_tls_handshake_certificate_request(handshake_view)
                elif handshake_type == TlsHandshakeMessageType.server_key_exchange:
                    handshake_data = _parse_tls_handshake_server_key_exchange(handshake_view, protocol_version)
                elif handshake_type == TlsHandshakeMessageType.client_key_exchange:
                    key_len = struct.unpack('B', handshake_view[:1])[0]
                    handshake_data = {'PublicKey': base64.b16encode(handshake_view[1:1 + key_len].tobytes()).decode()}
                formatted_handshake_data: typing.Dict[str, typing.Any] = {'HandshakeType': parse_enum(handshake_type)}
                if handshake_data is not None:
                    formatted_handshake_data['Data'] = handshake_data
                formatted_handshake_data['RawData'] = base64.b16encode(token_view[:4 + message_len].tobytes()).decode()
                data.append(formatted_handshake_data)
                token_view = token_view[4 + message_len:]
        formatted_data: typing.Dict[str, typing.Any] = {'ContentType': parse_enum(content_type), 'ProtocolVersion': parse_enum(protocol_version)}
        if data is not None:
            formatted_data['Data'] = data
        formatted_data['RawData'] = base64.b16encode(view[:5 + token_length].tobytes()).decode()
        res.append(formatted_data)
        view = view[5 + token_length:]
    return res