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
def authorize_url(self):
    """Returns the URL to start the OAuth dance.

        Must only be called from with a webapp.RequestHandler subclassed method
        that had been decorated with either @oauth_required or @oauth_aware.
        """
    url = self.flow.step1_get_authorize_url()
    return str(url)