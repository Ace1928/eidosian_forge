import os
import tempfile
import unittest
from webtest import TestApp
import pecan
from pecan.tests import PecanTestCase
def test_conf_changed(self):
    pecan.conf.server = pecan.configuration.Config({'port': '80'})
    assert pecan.conf.server.to_dict() == {'port': '80'}