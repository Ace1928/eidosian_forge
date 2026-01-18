import os
import pytest
import pyct.cmd
from pyct.cmd import fetch_data, clean_data, copy_examples, examples
def test_clean_data_when_data_file_is_real_does_nothing(tmp_project_with_test_file):
    project = tmp_project_with_test_file
    path = str(project / 'examples')
    data = project / 'examples' / 'data' / 'test_data.csv'
    data.write_text(REAL_FILE_CONTENT)
    clean_data(name='pyct', path=path)
    assert (project / 'examples' / 'data' / 'test_data.csv').is_file()
    assert (project / 'examples' / 'data' / 'test_data.csv').read_text() == REAL_FILE_CONTENT