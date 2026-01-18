import os
from pathlib import Path
from unittest import mock
import cirq_web
@mock.patch.dict(os.environ, {'BROWSER': 'true'})
def test_generate_html_file_with_browser(tmpdir):
    cirq_web.widget._DIST_PATH = Path(tmpdir) / 'dir'
    path = tmpdir.mkdir('dir')
    testfile_path = path.join('testfile.txt')
    testfile_path.write('This is a test bundle')
    test_widget = FakeWidget()
    test_html_path = test_widget.generate_html_file(str(path), 'test.html', open_in_browser=True)
    actual = open(test_html_path, 'r', encoding='utf-8').read()
    expected = f'\n        <meta charset="UTF-8">\n        <div id="{test_widget.id}"></div>\n        <script>This is a test bundle</script>\n        This is the test client code.\n        '
    assert remove_whitespace(expected) == remove_whitespace(actual)