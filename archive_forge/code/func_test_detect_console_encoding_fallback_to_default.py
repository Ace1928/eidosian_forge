import locale
import pytest
from pandas._config import detect_console_encoding
@pytest.mark.parametrize('std,locale', [['ascii', 'ascii'], ['ascii', locale.Error], [AttributeError, 'ascii'], [AttributeError, locale.Error], [OSError, 'ascii'], [OSError, locale.Error]])
def test_detect_console_encoding_fallback_to_default(monkeypatch, std, locale):
    with monkeypatch.context() as context:
        context.setattr('locale.getpreferredencoding', lambda: MockEncoding.raise_or_return(locale))
        context.setattr('sys.stdout', MockEncoding(std))
        context.setattr('sys.getdefaultencoding', lambda: 'sysDefaultEncoding')
        assert detect_console_encoding() == 'sysDefaultEncoding'