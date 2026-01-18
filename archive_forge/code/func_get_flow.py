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
def get_flow(self):
    """A thread local Flow object.

        Returns:
            A credentials.Flow object, or None if the flow hasn't been set in
            this thread yet, which happens in _create_flow() since Flows are
            created lazily.
        """
    return getattr(self._tls, 'flow', None)