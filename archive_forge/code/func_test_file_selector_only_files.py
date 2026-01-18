import os
import pytest
from panel.widgets import FileSelector
def test_file_selector_only_files(test_dir):
    selector = FileSelector(test_dir, only_files=True)
    selector._selector._lists[False].value = ['📁subdir1']
    selector._selector._buttons[True].clicks = 1
    assert selector.value == []
    assert selector._selector._lists[False].options == ['📁subdir1', '📁subdir2']