import sys
import unittest
from unittest import TestCase
from libcloud.compute.types import (
def test_compare_as_string(self):
    self.assertTrue(TestType.INUSE == 'inuse')
    self.assertFalse(TestType.INUSE == 'bar')