from __future__ import absolute_import, division, print_function
import io
import os
import json
import datetime
import importlib
from contextlib import redirect_stdout, suppress
from unittest import mock
import logging
from requests.models import Response, PreparedRequest
import pytest
from ansible.module_utils.six import raise_from
from awx.main.tests.functional.conftest import _request
from awx.main.tests.functional.conftest import credentialtype_scm, credentialtype_ssh  # noqa: F401; pylint: disable=unused-variable
from awx.main.models import (
from django.db import transaction
@pytest.fixture
def notification_template(organization):
    return NotificationTemplate.objects.create(name='test-notification_template', organization=organization, notification_type='webhook', notification_configuration=dict(url='http://localhost', username='', password='', headers={'Test': 'Header'}))