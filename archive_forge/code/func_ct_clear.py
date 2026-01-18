import re
import os_ken.exception
from os_ken.lib.ofctl_utils import str_to_int
from os_ken.ofproto import nicira_ext
@classmethod
def ct_clear(cls, ofproto, action_str):
    return dict(NXActionCTClear={})