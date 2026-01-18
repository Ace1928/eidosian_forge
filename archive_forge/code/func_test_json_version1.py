import json
from unittest import TestCase
import macaroonbakery.bakery as bakery
import macaroonbakery.checkers as checkers
import pymacaroons
import six
from macaroonbakery.tests import common
from pymacaroons import serializers
def test_json_version1(self):
    self._test_json_with_version(bakery.VERSION_1)