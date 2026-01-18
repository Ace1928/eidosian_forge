import copy
import json
import optparse
import os
import pickle
import sys
from urllib import parse
from troveclient.compat import client
from troveclient.compat import exceptions
def quote_user_host(user, host):
    quoted = ''
    if host:
        quoted = parse.quote('%s@%s' % (user, host))
    else:
        quoted = parse.quote('%s' % user)
    return quoted.replace('.', '%2e')