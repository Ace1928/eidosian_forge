import cgi
import json
import logging
import os
import pickle
import threading
from google.appengine.api import app_identity
from google.appengine.api import memcache
from google.appengine.api import users
from google.appengine.ext import db
from google.appengine.ext.webapp.util import login_required
import webapp2 as webapp
import oauth2client
from oauth2client import _helpers
from oauth2client import client
from oauth2client import clientsecrets
from oauth2client import transport
from oauth2client.contrib import xsrfutil
def callback_handler(self):
    """RequestHandler for the OAuth 2.0 redirect callback.

        Usage::

            app = webapp.WSGIApplication([
                ('/index', MyIndexHandler),
                ...,
                (decorator.callback_path, decorator.callback_handler())
            ])

        Returns:
            A webapp.RequestHandler that handles the redirect back from the
            server during the OAuth 2.0 dance.
        """
    decorator = self

    class OAuth2Handler(webapp.RequestHandler):
        """Handler for the redirect_uri of the OAuth 2.0 dance."""

        @login_required
        def get(self):
            error = self.request.get('error')
            if error:
                errormsg = self.request.get('error_description', error)
                self.response.out.write('The authorization request failed: {0}'.format(_safe_html(errormsg)))
            else:
                user = users.get_current_user()
                decorator._create_flow(self)
                credentials = decorator.flow.step2_exchange(self.request.params)
                decorator._storage_class(decorator._credentials_class, None, decorator._credentials_property_name, user=user).put(credentials)
                redirect_uri = _parse_state_value(str(self.request.get('state')), user)
                if redirect_uri is None:
                    self.response.out.write('The authorization request failed')
                    return
                if decorator._token_response_param and credentials.token_response:
                    resp_json = json.dumps(credentials.token_response)
                    redirect_uri = _helpers._add_query_parameter(redirect_uri, decorator._token_response_param, resp_json)
                self.redirect(redirect_uri)
    return OAuth2Handler