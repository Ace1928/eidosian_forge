import os
import pytest
import sys
import textwrap
import unittest
from contextlib import contextmanager
from traitlets.config.loader import Config
from IPython import get_ipython
from IPython.core import completer
from IPython.utils.tempdir import TemporaryDirectory, TemporaryWorkingDirectory
from IPython.utils.generics import complete_object
from IPython.testing import decorators as dec
from IPython.core.completer import (
def test_magic_config(self):
    ip = get_ipython()
    c = ip.Completer
    s, matches = c.complete(None, 'conf')
    self.assertIn('%config', matches)
    s, matches = c.complete(None, 'conf')
    self.assertNotIn('AliasManager', matches)
    s, matches = c.complete(None, 'config ')
    self.assertIn('AliasManager', matches)
    s, matches = c.complete(None, '%config ')
    self.assertIn('AliasManager', matches)
    s, matches = c.complete(None, 'config Ali')
    self.assertListEqual(['AliasManager'], matches)
    s, matches = c.complete(None, '%config Ali')
    self.assertListEqual(['AliasManager'], matches)
    s, matches = c.complete(None, 'config AliasManager')
    self.assertListEqual(['AliasManager'], matches)
    s, matches = c.complete(None, '%config AliasManager')
    self.assertListEqual(['AliasManager'], matches)
    s, matches = c.complete(None, 'config AliasManager.')
    self.assertIn('AliasManager.default_aliases', matches)
    s, matches = c.complete(None, '%config AliasManager.')
    self.assertIn('AliasManager.default_aliases', matches)
    s, matches = c.complete(None, 'config AliasManager.de')
    self.assertListEqual(['AliasManager.default_aliases'], matches)
    s, matches = c.complete(None, 'config AliasManager.de')
    self.assertListEqual(['AliasManager.default_aliases'], matches)