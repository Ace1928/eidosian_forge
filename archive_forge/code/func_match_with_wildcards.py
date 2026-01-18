import random
import re
import shelve
import ldap
from oslo_log import log
import keystone.conf
from keystone import exception
from keystone.identity.backends.ldap import common
def match_with_wildcards(norm_val, val_list):
    if norm_val.startswith('*'):
        if norm_val.endswith('*'):
            for x in val_list:
                if norm_val[1:-1] in x:
                    return True
        else:
            for x in val_list:
                if norm_val[1:] == x[len(x) - len(norm_val) + 1:]:
                    return True
    elif norm_val.endswith('*'):
        for x in val_list:
            if norm_val[:-1] == x[:len(norm_val) - 1]:
                return True
    else:
        for x in val_list:
            if check_value == x:
                return True
    return False