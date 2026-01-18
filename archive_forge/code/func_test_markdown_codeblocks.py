import ast
import runpy
from inspect import isclass
from pathlib import Path
import pytest
import panel as pn
@doc_available
@pytest.mark.parametrize('file', doc_files, ids=[str(f.relative_to(DOC_PATH)) for f in doc_files])
def test_markdown_codeblocks(file, tmp_path):
    from markdown_it import MarkdownIt
    exceptions = ('await', 'pn.serve', 'django', 'raise', 'display(')
    md_ast = MarkdownIt().parse(file.read_text(encoding='utf-8'))
    lines = ''
    for n in md_ast:
        if n.tag == 'code' and n.info is not None:
            if 'pyodide' in n.info.lower():
                if '>>>' not in n.content:
                    lines += n.content
    if not lines:
        return
    ast.parse(lines)
    if any((w in lines for w in exceptions)):
        return
    mod = tmp_path / f'{file.stem}.py'
    with open(mod, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    runpy.run_path(mod)