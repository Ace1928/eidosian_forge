import locale
import pytest
from pandas._config import detect_console_encoding
@pytest.mark.parametrize('encoding', [AttributeError, OSError, 'ascii'])
def test_detect_console_encoding_fallback_to_locale(monkeypatch, encoding):
    with monkeypatch.context() as context:
        context.setattr('locale.getpreferredencoding', lambda: 'foo')
        context.setattr('sys.stdout', MockEncoding(encoding))
        assert detect_console_encoding() == 'foo'