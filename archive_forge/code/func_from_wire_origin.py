import collections
from typing import Any, Callable, Iterator, List, Optional, Tuple, Union
import dns.exception
import dns.name
import dns.node
import dns.rdataclass
import dns.rdataset
import dns.rdatatype
import dns.rrset
import dns.serial
import dns.ttl
def from_wire_origin(self) -> Optional[dns.name.Name]:
    """Origin to use in from_wire() calls."""
    absolute_origin, relativize, _ = self.origin_information()
    if relativize:
        return absolute_origin
    else:
        return None