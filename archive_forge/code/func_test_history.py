import os
import re
from io import BytesIO, StringIO
import yaml
from .. import registry, tests, version_info_formats
from ..bzr.rio import read_stanzas
from ..version_info_formats.format_custom import (CustomVersionInfoBuilder,
from ..version_info_formats.format_python import PythonVersionInfoBuilder
from ..version_info_formats.format_rio import RioVersionInfoBuilder
from ..version_info_formats.format_yaml import YamlVersionInfoBuilder
from . import TestCaseWithTransport
def test_history(self):
    wt = self.create_branch()
    val = self.regen_text(wt, include_revision_history=True)
    self.assertContainsRe(val, 'id: r1')
    self.assertContainsRe(val, 'message: a')
    self.assertContainsRe(val, 'id: r2')
    self.assertContainsRe(val, 'message: ')
    self.assertContainsRe(val, 'id: r3')
    self.assertContainsRe(val, re.escape('message: "\\xE52"'))