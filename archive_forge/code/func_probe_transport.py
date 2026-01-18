from ... import version_info  # noqa: F401
from ... import controldir, errors
from ... import transport as _mod_transport
@classmethod
def probe_transport(klass, transport):
    try:
        external_url = transport.external_url()
    except errors.InProcessTransport:
        raise errors.NotBranchError(path=transport.base)
    scheme = external_url.split(':')[0]
    if scheme not in klass._supported_schemes:
        raise errors.NotBranchError(path=transport.base)
    from breezy import urlutils
    external_url = urlutils.strip_segment_parameters(external_url)
    if external_url.startswith('http:') or external_url.startswith('https:'):
        if klass._has_hg_http_smart_server(transport, external_url):
            return SmartHgDirFormat()
    raise errors.NotBranchError(path=transport.base)