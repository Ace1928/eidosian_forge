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
@_helpers.positional(2)
def oauth2decorator_from_clientsecrets(filename, scope, message=None, cache=None):
    """Creates an OAuth2Decorator populated from a clientsecrets file.

    Args:
        filename: string, File name of client secrets.
        scope: string or list of strings, scope(s) of the credentials being
               requested.
        message: string, A friendly string to display to the user if the
                 clientsecrets file is missing or invalid. The message may
                 contain HTML and will be presented on the web interface for
                 any method that uses the decorator.
        cache: An optional cache service client that implements get() and set()
               methods. See clientsecrets.loadfile() for details.

    Returns: An OAuth2Decorator
    """
    return OAuth2DecoratorFromClientSecrets(filename, scope, message=message, cache=cache)