from __future__ import absolute_import, unicode_literals
from oauthlib.common import extract_params, urlencode
from . import utils
def prepare_request_uri_query(oauth_params, uri):
    """Prepare the Request URI Query.

    Per `section 3.5.3`_ of the spec.

    .. _`section 3.5.3`: https://tools.ietf.org/html/rfc5849#section-3.5.3

    """
    sch, net, path, par, query, fra = urlparse(uri)
    query = urlencode(_append_params(oauth_params, extract_params(query) or []))
    return urlunparse((sch, net, path, par, query, fra))