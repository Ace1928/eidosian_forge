import requests
from ._compat import urljoin
def prepare_request(self, request, *args, **kwargs):
    """Prepare the request after generating the complete URL."""
    request.url = self.create_url(request.url)
    return super(BaseUrlSession, self).prepare_request(request, *args, **kwargs)