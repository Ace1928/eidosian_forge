import os
import pytest
import pyct.cmd
from pyct.cmd import fetch_data, clean_data, copy_examples, examples
def test_examples_with_prexisting_content_in_target_raises_error(tmp_project_with_examples):
    project = tmp_project_with_examples
    path = str(project / 'examples')
    data = project / 'examples' / 'data' / 'test_data.csv'
    data.write_text(REAL_FILE_CONTENT)
    with pytest.raises(ValueError):
        examples(name='pyct', path=path, use_test_data=True)
    assert (project / 'examples' / 'Test_Example_Notebook.ipynb').is_file()
    assert (project / 'examples' / 'Test_Example_Notebook.ipynb').read_text() != EXAMPLE_CONTENT
    assert (project / 'examples' / 'data' / 'test_data.csv').is_file()
    assert (project / 'examples' / 'data' / 'test_data.csv').read_text() == REAL_FILE_CONTENT