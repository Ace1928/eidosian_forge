import ast
import runpy
from inspect import isclass
from pathlib import Path
import pytest
import panel as pn
@doc_available
@pytest.mark.parametrize('doc_file', doc_files, ids=[str(f.relative_to(DOC_PATH)) for f in doc_files])
def test_markdown_indexed(doc_file):
    if str(doc_file).endswith('index.md') or doc_file.parent.name == 'examples':
        return
    index_page = doc_file.parent / 'index.md'
    filename = doc_file.name[:-3]
    if index_page.is_file():
        indexed = find_indexed(index_page)
        assert filename in indexed
    else:
        parent_name = doc_file.parent.name
        index_page = doc_file.parent.parent / f'{parent_name}.md'
        if not index_page.is_file():
            index_page = doc_file.parent.parent / 'index.md'
        indexed = find_indexed(index_page)
        assert f'{parent_name}/{filename}' in indexed