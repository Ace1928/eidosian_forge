import re
from functools import partial
from ovs.flow.flow import Flow, Section
from ovs.flow.kv import (
from ovs.flow.decoders import (
@staticmethod
def info_decoders():
    """Return the KVDecoders instance to parse the info section.

        Uses the cached version if available.
        """
    if not ODPFlow._info_decoders:
        ODPFlow._info_decoders = ODPFlow._gen_info_decoders()
    return ODPFlow._info_decoders