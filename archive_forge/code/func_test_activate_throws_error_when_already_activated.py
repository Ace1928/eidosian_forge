import os
import unittest
from unittest import mock
def test_activate_throws_error_when_already_activated(self):
    self.module.activated = True
    with self.assertRaises(ValueError):
        self.module.activate()