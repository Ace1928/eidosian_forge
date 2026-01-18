from .ssl_ import create_urllib3_context, resolve_cert_reqs, resolve_ssl_version
def create_proxy_ssl_context(ssl_version, cert_reqs, ca_certs=None, ca_cert_dir=None, ca_cert_data=None):
    """
    Generates a default proxy ssl context if one hasn't been provided by the
    user.
    """
    ssl_context = create_urllib3_context(ssl_version=resolve_ssl_version(ssl_version), cert_reqs=resolve_cert_reqs(cert_reqs))
    if not ca_certs and (not ca_cert_dir) and (not ca_cert_data) and hasattr(ssl_context, 'load_default_certs'):
        ssl_context.load_default_certs()
    return ssl_context