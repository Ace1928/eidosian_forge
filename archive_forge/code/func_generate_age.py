from __future__ import absolute_import, unicode_literals
import datetime
import os
from oauthlib.common import unicode_type, urldecode
def generate_age(issue_time):
    """Generate a age parameter for MAC authentication draft 00."""
    td = datetime.datetime.now() - issue_time
    age = (td.microseconds + (td.seconds + td.days * 24 * 3600) * 10 ** 6) / 10 ** 6
    return unicode_type(age)