import pytest
from urllib3.fields import RequestField, format_header_param_rfc2231, guess_content_type
from urllib3.packages.six import u
def test_make_multipart(self):
    field = RequestField('somename', 'data')
    field.make_multipart(content_type='image/jpg', content_location='/test')
    assert field.render_headers() == 'Content-Disposition: form-data; name="somename"\r\nContent-Type: image/jpg\r\nContent-Location: /test\r\n\r\n'