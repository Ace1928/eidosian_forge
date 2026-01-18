import os
import pytest
import pyct.cmd
from pyct.cmd import fetch_data, clean_data, copy_examples, examples
def test_clean_data_when_file_not_in_data_does_nothing(tmp_project_with_test_file):
    project = tmp_project_with_test_file
    path = str(project / 'examples')
    clean_data(name='pyct', path=path)
    assert not (project / 'examples' / 'data' / 'test_data.csv').is_file()
    assert (project / 'examples' / 'data' / '.data_stubs' / 'test_data.csv').is_file()
    assert (project / 'examples' / 'data' / '.data_stubs' / 'test_data.csv').read_text() == TEST_FILE_CONTENT