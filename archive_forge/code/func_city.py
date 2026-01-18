import socket
import geoip2.database
from django.conf import settings
from django.core.exceptions import ValidationError
from django.core.validators import validate_ipv46_address
from django.utils._os import to_path
from .resources import City, Country
def city(self, query):
    """
        Return a dictionary of city information for the given IP address or
        Fully Qualified Domain Name (FQDN). Some information in the dictionary
        may be undefined (None).
        """
    enc_query = self._check_query(query, city=True)
    return City(self._city.city(enc_query))