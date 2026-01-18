import os
import pytest
from panel.widgets import FileSelector
def test_file_selector_back_and_forward(test_dir):
    selector = FileSelector(test_dir)
    selector._directory.value = os.path.join(test_dir, 'subdir1')
    selector._go.clicks = 1
    assert selector._cwd == os.path.join(test_dir, 'subdir1')
    assert not selector._back.disabled
    assert selector._forward.disabled
    selector._back.clicks = 1
    assert selector._cwd == test_dir
    assert selector._back.disabled
    assert not selector._forward.disabled
    selector._forward.clicks = 1
    assert selector._cwd == os.path.join(test_dir, 'subdir1')