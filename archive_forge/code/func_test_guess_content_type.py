import pytest
from urllib3.fields import RequestField, format_header_param_rfc2231, guess_content_type
from urllib3.packages.six import u
@pytest.mark.parametrize('filename, content_types', [('image.jpg', ['image/jpeg', 'image/pjpeg']), ('notsure', ['application/octet-stream']), (None, ['application/octet-stream'])])
def test_guess_content_type(self, filename, content_types):
    assert guess_content_type(filename) in content_types