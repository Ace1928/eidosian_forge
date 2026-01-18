import sys
from struct import calcsize
from os_ken.lib import type_desc
from os_ken.ofproto.ofproto_common import OFP_HEADER_SIZE
from os_ken.ofproto import oxm_fields
def ofs_nbits(start, end):
    """
    The utility method for ofs_nbits

    This method is used in the class to set the ofs_nbits.

    This method converts start/end bits into ofs_nbits required to
    specify the bit range of OXM/NXM fields.

    ofs_nbits can be calculated as following::

      ofs_nbits = (start << 6) + (end - start)

    The parameter start/end  means the OXM/NXM field of ovs-ofctl command.

    ..
      field[start..end]
    ..

    +------------------------------------------+
    | *field*\\ **[**\\ *start*\\..\\ *end*\\ **]** |
    +------------------------------------------+

    ================ ======================================================
    Attribute        Description
    ================ ======================================================
    start            Start bit for OXM/NXM field
    end              End bit for OXM/NXM field
    ================ ======================================================
    """
    return (start << 6) + (end - start)