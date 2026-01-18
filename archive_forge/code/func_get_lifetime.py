import random
import re
import shelve
import ldap
from oslo_log import log
import keystone.conf
from keystone import exception
from keystone.identity.backends.ldap import common
def get_lifetime(self):
    return self._connection_time