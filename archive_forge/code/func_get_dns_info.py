import sys
import dns._features
def get_dns_info():
    """Extract resolver configuration."""
    getter = _getter_class()
    return getter.get()