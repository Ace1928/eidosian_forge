import os
import tempfile
import pytest
import requests
from panel.tests.util import (
import panel as pn
@linux_only
@pytest.mark.parametrize('relative', [True, False])
def test_custom_html_index(relative, html_file):
    index = '<html><body>Foo</body></html>'
    write_file(index, html_file.file)
    app = "import panel as pn; pn.Row('# Example').servable(title='A')"
    app2 = "import panel as pn; pn.Row('# Example 2').servable(title='B')"
    py1 = tempfile.NamedTemporaryFile(mode='w', suffix='.py')
    py2 = tempfile.NamedTemporaryFile(mode='w', suffix='.py')
    write_file(app, py1.file)
    write_file(app2, py2.file)
    if relative:
        index_path = os.path.basename(html_file.name)
        cwd = os.path.dirname(html_file.name)
    else:
        index_path = html_file.name
        cwd = None
    with run_panel_serve(['--port', '0', '--index', index_path, py1.name, py2.name], cwd=cwd) as p:
        port = wait_for_port(p.stdout)
        r = requests.get(f'http://localhost:{port}/')
        assert r.status_code == 200
        assert r.content.decode('utf-8') == index