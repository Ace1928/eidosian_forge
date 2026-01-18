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
def check_delete_name(self, check: CheckDeleteNameType) -> None:
    """Call *check* before putting (storing) an rdataset.

        The function is called with the transaction and the name.

        The check function may safely make non-mutating transaction method
        calls, but behavior is undefined if mutating transaction methods are
        called.  The check function should raise an exception if it objects to
        the put, and otherwise should return ``None``.
        """
    self._check_delete_name.append(check)