from django.contrib.sessions.backends.base import SessionBase
from django.core import signing
def cycle_key(self):
    """
        Keep the same data but with a new key. Call save() and it will
        automatically save a cookie with a new key at the end of the request.
        """
    self.save()