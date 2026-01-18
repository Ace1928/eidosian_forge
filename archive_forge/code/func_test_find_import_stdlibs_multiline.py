import pathlib
from panel.io.mime_render import (
def test_find_import_stdlibs_multiline():
    code = '\n    import re, io, time\n    '
    assert find_requirements(code) == []