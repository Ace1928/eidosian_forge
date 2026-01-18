import datetime
import json
import os
import socket
from tempfile import NamedTemporaryFile
import threading
import time
import sys
import google.auth
from google.auth import _helpers
from googleapiclient import discovery
from six.moves import BaseHTTPServer
from google.oauth2 import service_account
import pytest
from mock import patch
def get_xml_value_by_tagname(data, tagname):
    startIndex = data.index('<{}>'.format(tagname))
    if startIndex >= 0:
        endIndex = data.index('</{}>'.format(tagname), startIndex)
        if endIndex > startIndex:
            return data[startIndex + len(tagname) + 2:endIndex]