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
def test_custom_dotted_revno(self):
    sio = StringIO()
    wt = self.create_tree_with_dotted_revno()
    builder = CustomVersionInfoBuilder(wt.branch, working_tree=wt, template='{revno} revid: {revision_id}')
    builder.generate(sio)
    self.assertEqual('1.1.1 revid: o2', sio.getvalue())