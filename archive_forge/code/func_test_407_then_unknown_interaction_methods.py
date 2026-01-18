from unittest import TestCase
import macaroonbakery.httpbakery as httpbakery
import requests
from mock import patch
from httmock import HTTMock, response, urlmatch
def test_407_then_unknown_interaction_methods(self):

    class UnknownInteractor(httpbakery.Interactor):

        def kind(self):
            return 'unknown'
    client = httpbakery.Client(interaction_methods=[UnknownInteractor()])
    with HTTMock(first_407_then_200), HTTMock(discharge_401), HTTMock(visit_200):
        with self.assertRaises(httpbakery.InteractionError) as exc:
            requests.get(ID_PATH, cookies=client.cookies, auth=client.auth())
    self.assertEqual(str(exc.exception), 'cannot start interactive session: no methods supported; supported [unknown]; provided [interactive]')