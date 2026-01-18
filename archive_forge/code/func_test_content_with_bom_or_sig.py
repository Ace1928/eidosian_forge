from charset_normalizer.api import from_bytes
from charset_normalizer.models import CharsetMatches
import pytest
@pytest.mark.parametrize('payload, expected_encoding', [((u'\ufeff' + '我没有埋怨，磋砣的只是一些时间。').encode('gb18030'), 'gb18030'), ('我没有埋怨，磋砣的只是一些时间。'.encode('utf_32'), 'utf_32'), ('我没有埋怨，磋砣的只是一些时间。'.encode('utf_8_sig'), 'utf_8')])
def test_content_with_bom_or_sig(payload, expected_encoding):
    best_guess = from_bytes(payload).best()
    assert best_guess is not None, 'Detection but with SIG/BOM has failed!'
    assert best_guess.encoding == expected_encoding, 'Detection but with SIG/BOM is wrongly detected!'
    assert best_guess.byte_order_mark is True, 'The BOM/SIG property should return True'