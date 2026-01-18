import inspect
import struct
import base64
from . import packet_base
from . import ethernet
from os_ken import utils
from os_ken.lib.stringify import StringifyMixin
@classmethod
def from_jsondict(cls, dict_, decode_string=base64.b64decode, **additional_args):
    protocols = []
    for proto in dict_['protocols']:
        for key, value in proto.items():
            if key in PKT_CLS_DICT:
                pkt_cls = PKT_CLS_DICT[key]
                protocols.append(pkt_cls.from_jsondict(value))
            else:
                raise ValueError('unknown protocol name %s' % key)
    return cls(protocols=protocols)