import httplib2
import logging
import os
import sys
import time
from troveclient.compat import auth
from troveclient.compat import exceptions
def pretty_log(self, args, kwargs, resp, body):
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
    curl_cmd = ''.join(string_parts)
    LOG.debug('REQUEST:')
    if 'body' in kwargs:
        LOG.debug("%s -d '%s'", curl_cmd, kwargs['body'])
        try:
            req_body = json.dumps(json.loads(kwargs['body']), sort_keys=True, indent=4)
        except Exception:
            req_body = kwargs['body']
        LOG.debug('BODY: %s\n', req_body)
    else:
        LOG.debug(curl_cmd)
    try:
        resp_body = json.dumps(json.loads(body), sort_keys=True, indent=4)
    except Exception:
        resp_body = body
    LOG.debug('RESPONSE HEADERS: %s', resp)
    LOG.debug('RESPONSE BODY   : %s', resp_body)