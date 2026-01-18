import os
import pytest
from panel.widgets import FileSelector
def test_file_selector_multiple_across_dirs(test_dir):
    selector = FileSelector(test_dir)
    selector._selector._lists[False].value = ['📁subdir2']
    selector._selector._buttons[True].clicks = 1
    assert selector.value == [os.path.join(test_dir, 'subdir2')]
    selector._directory.value = os.path.join(test_dir, 'subdir1')
    selector._go.clicks = 1
    selector._selector._lists[False].value = ['a']
    selector._selector._buttons[True].clicks = 2
    assert selector.value == [os.path.join(test_dir, 'subdir2'), os.path.join(test_dir, 'subdir1', 'a')]
    selector._selector._lists[True].value = ['📁' + os.path.join('..', 'subdir2')]
    selector._selector._buttons[False].clicks = 1
    assert selector._selector.options == {'a': os.path.join(test_dir, 'subdir1', 'a'), 'b': os.path.join(test_dir, 'subdir1', 'b')}
    assert selector._selector._lists[False].options == ['b']
    assert selector.value == [os.path.join(test_dir, 'subdir1', 'a')]