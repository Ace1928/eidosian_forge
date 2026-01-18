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
def test_no_wt(self):
    wt = self.create_branch()
    self.build_tree(['branch/a', 'branch/c'])
    wt.add('c')
    wt.rename_one('b', 'd')
    bio = StringIO()
    builder = YamlVersionInfoBuilder(wt.branch, working_tree=None, check_for_clean=True, include_file_revisions=True, revision_id=None)
    builder.generate(bio)
    bio.seek(0)
    stanza = yaml.safe_load(bio)
    self.assertEqual([], stanza['file-revisions'])