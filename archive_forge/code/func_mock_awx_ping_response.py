from __future__ import absolute_import, division, print_function
import json
import sys
from awx.main.models import Organization, Team, Project, Inventory
from requests.models import Response
from unittest import mock
def mock_awx_ping_response(self, method, url, **kwargs):
    r = Response()
    r.getheader = getAWXheader.__get__(r)
    r.read = read.__get__(r)
    r.status = status.__get__(r)
    return r