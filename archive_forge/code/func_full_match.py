from typing import Any, Collection, Dict, Optional, Union, cast
import dns.name
import dns.rdataclass
import dns.rdataset
import dns.renderer
def full_match(self, name: dns.name.Name, rdclass: dns.rdataclass.RdataClass, rdtype: dns.rdatatype.RdataType, covers: dns.rdatatype.RdataType, deleting: Optional[dns.rdataclass.RdataClass]=None) -> bool:
    """Returns ``True`` if this rrset matches the specified name, class,
        type, covers, and deletion state.
        """
    if not super().match(rdclass, rdtype, covers):
        return False
    if self.name != name or self.deleting != deleting:
        return False
    return True