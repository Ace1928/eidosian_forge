import httplib2
import logging
import os
import sys
import time
from troveclient.compat import auth
from troveclient.compat import exceptions
def simple_log(self, args, kwargs, resp, body):
    if not LOG.isEnabledFor(logging.DEBUG):
        return
    string_parts = ['curl -i']
    for element in args:
        if element in ('GET', 'POST'):
            string_parts.append(' -X %s' % element)
        else:
            string_parts.append(' %s' % element)
    for element in kwargs['headers']:
        header = ' -H "%s: %s"' % (element, kwargs['headers'][element])
        string_parts.append(header)
    LOG.debug('REQ: %s\n', ''.join(string_parts))
    if 'body' in kwargs:
        LOG.debug('REQ BODY: %s\n', kwargs['body'])
    LOG.debug('RESP:%s %s\n', resp, body)