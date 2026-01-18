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
def isp(self, ip_address: IPAddress) -> ISP:
    """Get the ISP object for the IP address.

        :param ip_address: IPv4 or IPv6 address as a string.

        :returns: :py:class:`geoip2.models.ISP` object

        """
    return cast(ISP, self._flat_model_for(geoip2.models.ISP, 'GeoIP2-ISP', ip_address))