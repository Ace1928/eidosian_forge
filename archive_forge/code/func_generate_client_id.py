from __future__ import absolute_import, unicode_literals
import collections
import datetime
import logging
import re
import sys
import time
def generate_client_id(length=30, chars=CLIENT_ID_CHARACTER_SET):
    """Generates an OAuth client_id

    OAuth 2 specify the format of client_id in
    https://tools.ietf.org/html/rfc6749#appendix-A.
    """
    return generate_token(length, chars)