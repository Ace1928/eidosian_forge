import os
import pytest
import pyct.cmd
from pyct.cmd import fetch_data, clean_data, copy_examples, examples
def test_copy_examples(tmp_project):
    project = tmp_project
    path = str(project / 'examples')
    copy_examples(name='pyct', path=path)
    assert (project / 'examples' / 'Test_Example_Notebook.ipynb').is_file()