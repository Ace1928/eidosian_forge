import os
import tempfile
import unittest
from webtest import TestApp
import pecan
from pecan.tests import PecanTestCase
def test_conf_default(self):
    assert pecan.conf.server.to_dict() == {'port': '8080', 'host': '0.0.0.0'}