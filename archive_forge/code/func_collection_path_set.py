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
@pytest.fixture(autouse=True)
def collection_path_set(monkeypatch):
    """Monkey patch sys.path, insert the root of the collection folder
    so that content can be imported without being fully packaged
    """
    base_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
    monkeypatch.syspath_prepend(base_folder)