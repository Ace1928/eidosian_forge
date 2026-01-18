import os
import pytest
import pyct.cmd
from pyct.cmd import fetch_data, clean_data, copy_examples, examples
def test_copy_examples_with_prexisting_content_in_target_raises_error(tmp_project_with_examples):
    project = tmp_project_with_examples
    path = str(project / 'examples')
    with pytest.raises(ValueError):
        copy_examples(name='pyct', path=path)
    assert (project / 'examples' / 'Test_Example_Notebook.ipynb').is_file()
    assert (project / 'examples' / 'Test_Example_Notebook.ipynb').read_text() != EXAMPLE_CONTENT