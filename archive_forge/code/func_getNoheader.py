from __future__ import absolute_import, division, print_function
import json
import sys
from awx.main.models import Organization, Team, Project, Inventory
from requests.models import Response
from unittest import mock
def getNoheader(self, header_name, default):
    mock_headers = {}
    return mock_headers.get(header_name, default)