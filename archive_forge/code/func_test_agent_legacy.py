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
def test_agent_legacy(self):
    discharge_key = bakery.generate_key()

    class _DischargerLocator(bakery.ThirdPartyLocator):

        def third_party_info(self, loc):
            if loc == 'http://0.3.2.1':
                return bakery.ThirdPartyInfo(public_key=discharge_key.public_key, version=bakery.LATEST_VERSION)
    d = _DischargerLocator()
    server_key = bakery.generate_key()
    server_bakery = bakery.Bakery(key=server_key, locator=d)

    @urlmatch(path='.*/here')
    def server_get(url, request):
        ctx = checkers.AuthContext()
        test_ops = [bakery.Op(entity='test-op', action='read')]
        auth_checker = server_bakery.checker.auth(httpbakery.extract_macaroons(request.headers))
        try:
            auth_checker.allow(ctx, test_ops)
            resp = response(status_code=200, content='done')
        except bakery.PermissionDenied:
            caveats = [checkers.Caveat(location='http://0.3.2.1', condition='is-ok')]
            m = server_bakery.oven.macaroon(version=bakery.LATEST_VERSION, expiry=datetime.utcnow() + timedelta(days=1), caveats=caveats, ops=test_ops)
            content, headers = httpbakery.discharge_required_response(m, '/', 'test', 'message')
            resp = response(status_code=401, content=content, headers=headers)
        return request.hooks['response'][0](resp)

    class InfoStorage:
        info = None

    @urlmatch(path='.*/discharge')
    def discharge(url, request):
        qs = parse_qs(request.body)
        if qs.get('caveat64') is not None:
            content = {q: qs[q][0] for q in qs}

            class InteractionRequiredError(Exception):

                def __init__(self, error):
                    self.error = error

            class CheckerInError(bakery.ThirdPartyCaveatChecker):

                def check_third_party_caveat(self, ctx, info):
                    InfoStorage.info = info
                    raise InteractionRequiredError(httpbakery.Error(code=httpbakery.ERR_INTERACTION_REQUIRED, version=httpbakery.request_version(request.headers), message='interaction required', info=httpbakery.ErrorInfo(wait_url='http://0.3.2.1/wait?dischargeid=1', visit_url='http://0.3.2.1/visit?dischargeid=1')))
            try:
                httpbakery.discharge(checkers.AuthContext(), content, discharge_key, None, CheckerInError())
            except InteractionRequiredError as exc:
                return response(status_code=401, content={'Code': exc.error.code, 'Message': exc.error.message, 'Info': {'WaitURL': exc.error.info.wait_url, 'VisitURL': exc.error.info.visit_url}}, headers={'Content-Type': 'application/json'})
    key = bakery.generate_key()

    @urlmatch(path='.*/visit')
    def visit(url, request):
        if request.headers.get('Accept') == 'application/json':
            return {'status_code': 200, 'content': {'agent': '/agent-visit'}}
        raise Exception('unexpected call to visit without Accept header')

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

    @urlmatch(path='.*/wait$')
    def wait(url, request):

        class EmptyChecker(bakery.ThirdPartyCaveatChecker):

            def check_third_party_caveat(self, ctx, info):
                return []
        if InfoStorage.info is None:
            self.fail('visit url has not been visited')
        m = bakery.discharge(checkers.AuthContext(), InfoStorage.info.id, InfoStorage.info.caveat, discharge_key, EmptyChecker(), _DischargerLocator())
        return {'status_code': 200, 'content': {'Macaroon': m.to_dict()}}
    with HTTMock(server_get), HTTMock(discharge), HTTMock(visit), HTTMock(wait), HTTMock(agent_visit):
        client = httpbakery.Client(interaction_methods=[agent.AgentInteractor(agent.AuthInfo(key=key, agents=[agent.Agent(username='test-user', url=u'http://0.3.2.1')]))])
        resp = requests.get('http://0.1.2.3/here', cookies=client.cookies, auth=client.auth())
    self.assertEqual(resp.content, b'done')