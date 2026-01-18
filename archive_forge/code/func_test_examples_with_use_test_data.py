import os
import pytest
import pyct.cmd
from pyct.cmd import fetch_data, clean_data, copy_examples, examples
def test_examples_with_use_test_data(tmp_project):
    project = tmp_project
    path = str(project / 'examples')
    examples(name='pyct', path=path, use_test_data=True)
    assert (project / 'examples' / 'data' / 'test_data.csv').is_file()
    assert (project / 'examples' / 'Test_Example_Notebook.ipynb').is_file()