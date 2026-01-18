import itertools
from oslo_log import log
import passlib.hash
import keystone.conf
from keystone import exception
from keystone.i18n import _
def hash_user_password(user):
    """Hash a user dict's password without modifying the passed-in dict."""
    password = user.get('password')
    if password is None:
        return user
    return dict(user, password=hash_password(password))