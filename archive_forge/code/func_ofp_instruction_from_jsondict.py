import base64
import collections
import logging
import struct
import functools
from os_ken import exception
from os_ken import utils
from os_ken.lib import stringify
from os_ken.ofproto import ofproto_common
def ofp_instruction_from_jsondict(dp, jsonlist, encap=True):
    """
    This function is intended to be used with
    os_ken.lib.ofctl_string.ofp_instruction_from_str.
    It is very similar to ofp_msg_from_jsondict, but works on
    a list of OFPInstructions/OFPActions. It also encapsulates
    OFPAction into OFPInstructionActions, as >OF1.0 OFPFlowMod
    requires that.

    This function takes the following arguments.

    ======== ==================================================
    Argument Description
    ======== ==================================================
    dp       An instance of os_ken.controller.Datapath.
    jsonlist A list of JSON style dictionaries.
    encap    Encapsulate OFPAction into OFPInstructionActions.
             Must be false for OF10.
    ======== ==================================================
    """
    proto = dp.ofproto
    parser = dp.ofproto_parser
    actions = []
    result = []
    for jsondict in jsonlist:
        assert len(jsondict) == 1
        k, v = list(jsondict.items())[0]
        cls = getattr(parser, k)
        if issubclass(cls, parser.OFPAction):
            if encap:
                actions.append(cls.from_jsondict(v))
                continue
        else:
            ofpinst = getattr(parser, 'OFPInstruction', None)
            if not ofpinst or not issubclass(cls, ofpinst):
                raise ValueError('Supplied jsondict is of wrong type: %s', jsondict)
        result.append(cls.from_jsondict(v))
    if not encap:
        return result
    if actions:
        result = [parser.OFPInstructionActions(proto.OFPIT_APPLY_ACTIONS, actions)] + result
    return result