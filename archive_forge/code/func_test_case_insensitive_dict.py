import datetime
import math
import os
import random
import re
import subprocess
import sys
import time
import types
import unittest
import warnings
from humanfriendly import (
from humanfriendly.case import CaseInsensitiveDict, CaseInsensitiveKey
from humanfriendly.cli import main
from humanfriendly.compat import StringIO
from humanfriendly.decorators import cached
from humanfriendly.deprecation import DeprecationProxy, define_aliases, deprecated_args, get_aliases
from humanfriendly.prompts import (
from humanfriendly.sphinx import (
from humanfriendly.tables import (
from humanfriendly.terminal import (
from humanfriendly.terminal.html import html_to_ansi
from humanfriendly.terminal.spinners import AutomaticSpinner, Spinner
from humanfriendly.testing import (
from humanfriendly.text import (
from humanfriendly.usage import (
from mock import MagicMock
def test_case_insensitive_dict(self):
    """Test the CaseInsensitiveDict class."""
    assert len(CaseInsensitiveDict([('key', True), ('KEY', False)])) == 1
    assert len(CaseInsensitiveDict([('one', True), ('ONE', False)], one=False, two=True)) == 2
    assert len(CaseInsensitiveDict(dict(key=True, KEY=False))) == 1
    assert len(CaseInsensitiveDict(dict(one=True, ONE=False), one=False, two=True)) == 2
    assert len(CaseInsensitiveDict(one=True, ONE=False, two=True)) == 2
    obj = CaseInsensitiveDict.fromkeys(['One', 'one', 'ONE', 'Two', 'two', 'TWO'])
    assert len(obj) == 2
    obj = CaseInsensitiveDict(existing_key=42)
    assert obj.get('Existing_Key') == 42
    obj = CaseInsensitiveDict(existing_key=42)
    assert obj.pop('Existing_Key') == 42
    assert len(obj) == 0
    obj = CaseInsensitiveDict(existing_key=42)
    assert obj.setdefault('Existing_Key') == 42
    obj.setdefault('other_key', 11)
    assert obj['Other_Key'] == 11
    obj = CaseInsensitiveDict(existing_key=42)
    assert 'Existing_Key' in obj
    obj = CaseInsensitiveDict(existing_key=42)
    del obj['Existing_Key']
    assert len(obj) == 0
    obj = CaseInsensitiveDict(existing_key=42)
    assert obj['Existing_Key'] == 42
    obj = CaseInsensitiveDict(existing_key=42)
    obj['Existing_Key'] = 11
    assert obj['existing_key'] == 11