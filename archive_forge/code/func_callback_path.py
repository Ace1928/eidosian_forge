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
@property
def callback_path(self):
    """The absolute path where the callback will occur.

        Note this is the absolute path, not the absolute URI, that will be
        calculated by the decorator at runtime. See callback_handler() for how
        this should be used.

        Returns:
            The callback path as a string.
        """
    return self._callback_path