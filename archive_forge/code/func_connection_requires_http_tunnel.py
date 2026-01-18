from .ssl_ import create_urllib3_context, resolve_cert_reqs, resolve_ssl_version
def connection_requires_http_tunnel(proxy_url=None, proxy_config=None, destination_scheme=None):
    """
    Returns True if the connection requires an HTTP CONNECT through the proxy.

    :param URL proxy_url:
        URL of the proxy.
    :param ProxyConfig proxy_config:
        Proxy configuration from poolmanager.py
    :param str destination_scheme:
        The scheme of the destination. (i.e https, http, etc)
    """
    if proxy_url is None:
        return False
    if destination_scheme == 'http':
        return False
    if proxy_url.scheme == 'https' and proxy_config and proxy_config.use_forwarding_for_https:
        return False
    return True