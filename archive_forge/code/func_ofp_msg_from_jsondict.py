import base64
import collections
import logging
import struct
import functools
from os_ken import exception
from os_ken import utils
from os_ken.lib import stringify
from os_ken.ofproto import ofproto_common
def ofp_msg_from_jsondict(dp, jsondict):
    """
    This function instanticates an appropriate OpenFlow message class
    from the given JSON style dictionary.
    The objects created by following two code fragments are equivalent.

    Code A::

        jsonstr = '{ "OFPSetConfig": { "flags": 0, "miss_send_len": 128 } }'
        jsondict = json.loads(jsonstr)
        o = ofp_msg_from_jsondict(dp, jsondict)

    Code B::

        o = dp.ofproto_parser.OFPSetConfig(flags=0, miss_send_len=128)

    This function takes the following arguments.

    ======== =======================================
    Argument Description
    ======== =======================================
    dp       An instance of os_ken.controller.Datapath.
    jsondict A JSON style dict.
    ======== =======================================
    """
    parser = dp.ofproto_parser
    assert len(jsondict) == 1
    for k, v in jsondict.items():
        cls = getattr(parser, k)
        assert issubclass(cls, MsgBase)
        return cls.from_jsondict(v, datapath=dp)