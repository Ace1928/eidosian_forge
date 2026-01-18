import socket
import geoip2.database
from django.conf import settings
from django.core.exceptions import ValidationError
from django.core.validators import validate_ipv46_address
from django.utils._os import to_path
from .resources import City, Country
def country_name(self, query):
    """Return the country name for the given IP Address or FQDN."""
    return self.country(query)['country_name']