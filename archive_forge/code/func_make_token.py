from datetime import datetime
from django.conf import settings
from django.utils.crypto import constant_time_compare, salted_hmac
from django.utils.http import base36_to_int, int_to_base36
def make_token(self, user):
    """
        Return a token that can be used once to do a password reset
        for the given user.
        """
    return self._make_token_with_timestamp(user, self._num_seconds(self._now()), self.secret)