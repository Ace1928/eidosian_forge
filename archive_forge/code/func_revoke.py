import collections
import copy
import datetime
import json
import logging
import os
import shutil
import socket
import sys
import tempfile
import six
from six.moves import http_client
from six.moves import urllib
import oauth2client
from oauth2client import _helpers
from oauth2client import _pkce
from oauth2client import clientsecrets
from oauth2client import transport
def revoke(self, http):
    """Revokes a refresh_token and makes the credentials void.

        Args:
            http: httplib2.Http, an http object to be used to make the revoke
                  request.
        """
    self._revoke(http)