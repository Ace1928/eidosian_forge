from __future__ import absolute_import, division, print_function
import json
import sys
from awx.main.models import Organization, Team, Project, Inventory
from requests.models import Response
from unittest import mock
def getTowerheader(self, header_name, default):
    mock_headers = {'X-API-Product-Name': controller_name, 'X-API-Product-Version': ping_version}
    return mock_headers.get(header_name, default)