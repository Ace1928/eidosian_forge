import numbers
import re
import socket
from os_ken.lib import ip
def is_valid_mpls_label(label):
    """Validates `label` according to MPLS label rules

    RFC says:
    This 20-bit field.
    A value of 0 represents the "IPv4 Explicit NULL Label".
    A value of 1 represents the "Router Alert Label".
    A value of 2 represents the "IPv6 Explicit NULL Label".
    A value of 3 represents the "Implicit NULL Label".
    Values 4-15 are reserved.
    """
    if not isinstance(label, numbers.Integral) or 4 <= label <= 15 or (label < 0 or label > 2 ** 20):
        return False
    return True