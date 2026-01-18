import json
import logging
import os
import tempfile
from datetime import datetime, timedelta
from unittest import TestCase
import macaroonbakery.bakery as bakery
import macaroonbakery.checkers as checkers
import macaroonbakery.httpbakery as httpbakery
import macaroonbakery.httpbakery.agent as agent
import requests.cookies
from httmock import HTTMock, response, urlmatch
from six.moves.urllib.parse import parse_qs, urlparse
@urlmatch(path='.*/agent-visit')
def agent_visit(url, request):
    if request.method != 'POST':
        raise Exception('unexpected method')
    log.info('agent_visit url {}'.format(url))
    body = json.loads(request.body.decode('utf-8'))
    if body['username'] != 'test-user':
        raise Exception('unexpected username in body {!r}'.format(request.body))
    public_key = bakery.PublicKey.deserialize(body['public_key'])
    ms = httpbakery.extract_macaroons(request.headers)
    if len(ms) == 0:
        b = bakery.Bakery(key=discharge_key)
        m = b.oven.macaroon(version=bakery.LATEST_VERSION, expiry=datetime.utcnow() + timedelta(days=1), caveats=[bakery.local_third_party_caveat(public_key, version=httpbakery.request_version(request.headers))], ops=[bakery.Op(entity='agent', action='login')])
        content, headers = httpbakery.discharge_required_response(m, '/', 'test', 'message')
        resp = response(status_code=401, content=content, headers=headers)
        return request.hooks['response'][0](resp)
    return {'status_code': 200, 'content': {'agent_login': True}}