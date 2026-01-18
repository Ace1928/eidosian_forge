import locale
import pytest
from pandas._config import detect_console_encoding
@pytest.mark.parametrize('empty,filled', [['stdin', 'stdout'], ['stdout', 'stdin']])
def test_detect_console_encoding_from_stdout_stdin(monkeypatch, empty, filled):
    with monkeypatch.context() as context:
        context.setattr(f'sys.{empty}', MockEncoding(''))
        context.setattr(f'sys.{filled}', MockEncoding(filled))
        assert detect_console_encoding() == filled