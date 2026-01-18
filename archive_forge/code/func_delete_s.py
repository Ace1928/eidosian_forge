import random
import re
import shelve
import ldap
from oslo_log import log
import keystone.conf
from keystone import exception
from keystone.identity.backends.ldap import common
def delete_s(self, dn):
    """Remove the ldap object at specified dn."""
    return self.delete_ext_s(dn, serverctrls=[])