import os
import pytest
import pyct.cmd
from pyct.cmd import fetch_data, clean_data, copy_examples, examples
def test_clean_data_when_stubs_is_empty_does_nothing(tmp_project_with_stubs):
    project = tmp_project_with_stubs
    path = str(project / 'examples')
    data = project / 'examples' / 'data' / 'test_data.csv'
    data.write_text(REAL_FILE_CONTENT)
    clean_data(name='pyct', path=path)
    assert (project / 'examples' / 'data' / 'test_data.csv').is_file()
    assert not (project / 'examples' / 'data' / '.data_stubs' / 'test_data.csv').is_file()