import functools
@property
def exploded(self):
    """Return the longhand version of the IP address as a string."""
    return self._explode_shorthand_ip_string()