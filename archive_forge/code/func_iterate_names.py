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
def iterate_names(self) -> Iterator[dns.name.Name]:
    """Iterate all the names in the transaction.

        Note that as is usual with python iterators, adding or removing names
        while iterating will invalidate the iterator and may raise `RuntimeError`
        or fail to iterate over all entries."""
    self._check_ended()
    return self._iterate_names()