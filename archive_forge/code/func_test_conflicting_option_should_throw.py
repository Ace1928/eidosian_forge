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
def test_conflicting_option_should_throw(self):

    class MyApp(application.App):

        def __init__(self):
            super(MyApp, self).__init__(description='testing', version='0.1', command_manager=commandmanager.CommandManager('tests'))

        def build_option_parser(self, description, version):
            parser = super(MyApp, self).build_option_parser(description, version)
            parser.add_argument('-h', '--help', default=self, help='Show help message and exit.')
    self.assertRaises(argparse.ArgumentError, MyApp)