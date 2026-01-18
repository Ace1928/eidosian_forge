import os
import urllib.parse
import uuid
from lxml import etree
from oslo_config import fixture as config
import requests
from keystoneclient.auth import conf
from keystoneclient.contrib.auth.v3 import saml2
from keystoneclient import exceptions
from keystoneclient import session
from keystoneclient.tests.unit.v3 import client_fixtures
from keystoneclient.tests.unit.v3 import saml2_fixtures
from keystoneclient.tests.unit.v3 import utils
from keystoneclient.v3.contrib.federation import saml as saml_manager
def test_custom_303_redirection(self):
    self.requests_mock.post(self.SHIB_CONSUMER_URL, text='BODY', headers={'location': self.FEDERATION_AUTH_URL}, status_code=303)
    self.requests_mock.get(self.FEDERATION_AUTH_URL, json=saml2_fixtures.UNSCOPED_TOKEN, headers={'X-Subject-Token': saml2_fixtures.UNSCOPED_TOKEN_HEADER})
    self.session.redirect = False
    response = self.session.post(self.SHIB_CONSUMER_URL, data='CLIENT BODY')
    self.assertEqual(303, response.status_code)
    self.assertEqual(self.FEDERATION_AUTH_URL, response.headers['location'])
    response = self.saml2plugin._handle_http_ecp_redirect(self.session, response, 'GET')
    self.assertEqual(self.FEDERATION_AUTH_URL, response.request.url)
    self.assertEqual('GET', response.request.method)