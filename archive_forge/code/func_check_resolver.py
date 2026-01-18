from collections import Counter
from django.conf import settings
from . import Error, Tags, Warning, register
def check_resolver(resolver):
    """
    Recursively check the resolver.
    """
    check_method = getattr(resolver, 'check', None)
    if check_method is not None:
        return check_method()
    elif not hasattr(resolver, 'resolve'):
        return get_warning_for_invalid_pattern(resolver)
    else:
        return []