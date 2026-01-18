import pathlib
from panel.io.mime_render import (
def test_find_import_replacement():
    code = '\n    import transformers_js\n    '
    assert find_requirements(code) == ['transformers-js-py']