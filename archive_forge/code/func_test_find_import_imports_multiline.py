import pathlib
from panel.io.mime_render import (
def test_find_import_imports_multiline():
    code = '\n    import numpy, scipy\n    '
    assert find_requirements(code) == ['numpy', 'scipy']