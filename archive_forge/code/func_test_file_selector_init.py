import os
import pytest
from panel.widgets import FileSelector
def test_file_selector_init(test_dir):
    selector = FileSelector(test_dir)
    assert selector._selector.options == {'📁subdir1': os.path.join(test_dir, 'subdir1'), '📁subdir2': os.path.join(test_dir, 'subdir2')}