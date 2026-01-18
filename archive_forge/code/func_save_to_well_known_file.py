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
def save_to_well_known_file(credentials, well_known_file=None):
    """Save the provided GoogleCredentials to the well known file.

    Args:
        credentials: the credentials to be saved to the well known file;
                     it should be an instance of GoogleCredentials
        well_known_file: the name of the file where the credentials are to be
                         saved; this parameter is supposed to be used for
                         testing only
    """
    if well_known_file is None:
        well_known_file = _get_well_known_file()
    config_dir = os.path.dirname(well_known_file)
    if not os.path.isdir(config_dir):
        raise OSError('Config directory does not exist: {0}'.format(config_dir))
    credentials_data = credentials.serialization_data
    _save_private_file(well_known_file, credentials_data)