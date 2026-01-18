from collections import Counter
from django.conf import settings
from . import Error, Tags, Warning, register
def get_warning_for_invalid_pattern(pattern):
    """
    Return a list containing a warning that the pattern is invalid.

    describe_pattern() cannot be used here, because we cannot rely on the
    urlpattern having regex or name attributes.
    """
    if isinstance(pattern, str):
        hint = "Try removing the string '{}'. The list of urlpatterns should not have a prefix string as the first element.".format(pattern)
    elif isinstance(pattern, tuple):
        hint = 'Try using path() instead of a tuple.'
    else:
        hint = None
    return [Error('Your URL pattern {!r} is invalid. Ensure that urlpatterns is a list of path() and/or re_path() instances.'.format(pattern), hint=hint, id='urls.E004')]