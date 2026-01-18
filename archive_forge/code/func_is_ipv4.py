import ipaddress
import time
from datetime import datetime
from enum import Enum
def is_ipv4(ip):
    try:
        ipaddress.IPv4Address(ip)
        return True
    except ipaddress.AddressValueError:
        return False