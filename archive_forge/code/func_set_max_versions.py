import collections
import threading
from typing import Callable, Deque, Optional, Set, Union
import dns.exception
import dns.immutable
import dns.name
import dns.node
import dns.rdataclass
import dns.rdataset
import dns.rdatatype
import dns.rdtypes.ANY.SOA
import dns.zone
def set_max_versions(self, max_versions: Optional[int]) -> None:
    """Set a pruning policy that retains up to the specified number
        of versions
        """
    if max_versions is not None and max_versions < 1:
        raise ValueError('max versions must be at least 1')
    if max_versions is None:

        def policy(zone, _):
            return False
    else:

        def policy(zone, _):
            return len(zone._versions) > max_versions
    self.set_pruning_policy(policy)