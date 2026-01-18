import datetime
import logging
from cheroot.test import webtest
import pytest
import requests  # FIXME: Temporary using it directly, better switch
import cherrypy
from cherrypy.test.logtest import LogCase
@pytest.fixture
def error_log_file(tmp_path_factory):
    return tmp_path_factory.mktemp('logs') / 'access.log'