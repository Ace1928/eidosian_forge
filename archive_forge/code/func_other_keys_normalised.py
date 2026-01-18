import email.utils
import logging
import os
import re
import socket
from debian.debian_support import Version
def other_keys_normalised(self):
    """ Obtain a dict from the block header (other than urgency) """
    norm_dict = {}
    for key, value in self.other_pairs.items():
        key = key[0].upper() + key[1:].lower()
        m = xbcs_re.match(key)
        if m is None:
            key = 'XS-%s' % key
        norm_dict[key] = value
    return norm_dict