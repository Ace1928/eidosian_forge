from __future__ import annotations
import ipaddress
import random
from typing import Any, Optional, Union
from pymongo.common import CONNECT_TIMEOUT
from pymongo.errors import ConfigurationError
def get_hosts_and_min_ttl(self) -> tuple[list[tuple[str, Any]], int]:
    results, nodes = self._get_srv_response_and_hosts(False)
    rrset = results.rrset
    ttl = rrset.ttl if rrset else 0
    return (nodes, ttl)