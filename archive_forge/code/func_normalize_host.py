import re
from . import compat
from . import misc
def normalize_host(host):
    """Normalize a host string."""
    if misc.IPv6_MATCHER.match(host):
        percent = host.find('%')
        if percent != -1:
            percent_25 = host.find('%25')
            if percent_25 == -1 or percent < percent_25 or (percent == percent_25 and percent_25 == len(host) - 4):
                host = host.replace('%', '%25', 1)
            return host[:percent].lower() + host[percent:]
    return host.lower()