import ipaddress
import json
from typing import Any, Dict, cast, List, Optional, Type, Union
import aiohttp
import aiohttp.http
import requests
import requests.utils
import geoip2
import geoip2.models
from geoip2.errors import (
from geoip2.models import City, Country, Insights
from geoip2.types import IPAddress
def insights(self, ip_address: IPAddress='me') -> Insights:
    """Call the Insights endpoint with the specified IP.

        Insights is only supported by the GeoIP2 web service. The GeoLite2 web
        service does not support it.

        :param ip_address: IPv4 or IPv6 address as a string. If no address
          is provided, the address that the web service is called from will
          be used.

        :returns: :py:class:`geoip2.models.Insights` object

        """
    return cast(Insights, self._response_for('insights', geoip2.models.Insights, ip_address))