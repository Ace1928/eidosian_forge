from __future__ import absolute_import, unicode_literals
import re
from abc import ABCMeta, abstractmethod
from unittest import TestCase
import pytest
import six
from pybtex import textutils
from pybtex.richtext import HRef, Protected, String, Symbol, Tag, Text, nbsp
def test_add_period(self):
    assert nbsp.add_period().render_as('html') == '&nbsp;.'
    assert nbsp.add_period().add_period().render_as('html') == '&nbsp;.'