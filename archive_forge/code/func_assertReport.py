import os
from io import StringIO
from .. import delta as _mod_delta
from .. import revision as _mod_revision
from .. import tests
from ..bzr.inventorytree import InventoryTreeChange
def assertReport(self, expected, file_id=b'fid', path='path', versioned_change='unchanged', renamed=False, copied=False, modified='unchanged', exe_change=False, kind=('file', 'file'), old_path=None, unversioned_filter=None, view_info=None):
    if expected is None:
        expected_lines = None
    else:
        expected_lines = [expected]
    self.assertReportLines(expected_lines, file_id, path, versioned_change, renamed, copied, modified, exe_change, kind, old_path, unversioned_filter, view_info)