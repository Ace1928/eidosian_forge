import inspect
import os
from typing import Any, AnyStr, cast, IO, List, Optional, Type, Union
import maxminddb
from maxminddb import (
import geoip2
import geoip2.models
import geoip2.errors
from geoip2.types import IPAddress
from geoip2.models import (
def connection_type(self, ip_address: IPAddress) -> ConnectionType:
    """Get the ConnectionType object for the IP address.

        :param ip_address: IPv4 or IPv6 address as a string.

        :returns: :py:class:`geoip2.models.ConnectionType` object

        """
    return cast(ConnectionType, self._flat_model_for(geoip2.models.ConnectionType, 'GeoIP2-Connection-Type', ip_address))