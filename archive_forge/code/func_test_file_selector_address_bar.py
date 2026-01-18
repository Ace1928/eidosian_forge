import os
import pytest
from panel.widgets import FileSelector
def test_file_selector_address_bar(test_dir):
    selector = FileSelector(test_dir)
    selector._directory.value = os.path.join(test_dir, 'subdir1')
    assert not selector._go.disabled
    selector._go.clicks = 1
    assert selector._cwd == os.path.join(test_dir, 'subdir1')
    assert selector._go.disabled
    assert selector._forward.disabled
    assert not selector._back.disabled
    assert selector._selector.options == {'a': os.path.join(test_dir, 'subdir1', 'a'), 'b': os.path.join(test_dir, 'subdir1', 'b')}
    selector._up.clicks = 1
    selector._selector._lists[False].value = ['subdir1']
    assert selector._directory.value == os.path.join(test_dir, 'subdir1')
    selector._selector._lists[False].value = []
    assert selector._directory.value == test_dir