import collections
import http.client as http
import io
from unittest import mock
import copy
import os
import sys
import uuid
import fixtures
from oslo_serialization import jsonutils
import webob
from glance.cmd import replicator as glance_replicator
from glance.common import exception
from glance.tests.unit import utils as unit_test_utils
from glance.tests import utils as test_utils
def prime_request(self, method, url, in_body, in_headers, out_code, out_body, out_headers):
    if not url.startswith('/'):
        url = '/' + url
    url = unit_test_utils.sort_url_by_qs_keys(url)
    hkeys = sorted(in_headers.keys())
    hashable = (method, url, in_body, ' '.join(hkeys))
    flat_headers = []
    for key in out_headers:
        flat_headers.append((key, out_headers[key]))
    self.reqs[hashable] = (out_code, out_body, flat_headers)