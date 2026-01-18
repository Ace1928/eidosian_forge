from io import StringIO
import pytest
from panel.io.handlers import capture_code_cell, extract_code, parse_notebook
import panel as pn
import panel as pn
import panel as pn
import panel as pn
import panel"""
def test_extract_description_block():
    f = StringIO(md3)
    assert extract_code(f) == "import panel as pn\n\npn.extension()\n\npn.pane.Markdown('\\nMy description\\n\\n').servable()\n\npn.Row(1, 2, 3).servable()\n"