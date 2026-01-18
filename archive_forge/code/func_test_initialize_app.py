import argparse
import codecs
import io
from unittest import mock
from cliff import app as application
from cliff import command as c_cmd
from cliff import commandmanager
from cliff.tests import base
from cliff.tests import utils as test_utils
from cliff import utils
import sys
def test_initialize_app(self):
    app, command = make_app()
    app.initialize_app = mock.MagicMock(name='initialize_app')
    app.run(['mock'])
    app.initialize_app.assert_called_once_with(['mock'])