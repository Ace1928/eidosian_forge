import functools
import json
import os
import ssl
import subprocess
import sys
import threading
import time
import traceback
import http.client
import OpenSSL.SSL
import pytest
import requests
import trustme
from .._compat import bton, ntob, ntou
from .._compat import IS_ABOVE_OPENSSL10, IS_CI, IS_PYPY
from .._compat import IS_LINUX, IS_MACOS, IS_WINDOWS
from ..server import HTTPServer, get_ssl_adapter_class
from ..testing import (
from ..wsgi import Gateway_10
@pytest.mark.parametrize('adapter_type', ('builtin', 'pyopenssl'))
@pytest.mark.parametrize('ip_addr', (ANY_INTERFACE_IPV4, pytest.param(ANY_INTERFACE_IPV6, marks=missing_ipv6)))
@pytest.mark.flaky(reruns=3, reruns_delay=2)
def test_http_over_https_error(http_request_timeout, tls_http_server, adapter_type, ca, ip_addr, tls_certificate, tls_certificate_chain_pem_path, tls_certificate_private_key_pem_path):
    """Ensure that connecting over HTTP to HTTPS port is handled."""
    issue_225 = IS_MACOS and adapter_type == 'builtin'
    if issue_225:
        pytest.xfail('Test fails in Travis-CI')
    tls_adapter_cls = get_ssl_adapter_class(name=adapter_type)
    tls_adapter = tls_adapter_cls(tls_certificate_chain_pem_path, tls_certificate_private_key_pem_path)
    if adapter_type == 'pyopenssl':
        tls_adapter.context = tls_adapter.get_context()
    tls_certificate.configure_cert(tls_adapter.context)
    interface, _host, port = _get_conn_data(ip_addr)
    tlshttpserver = tls_http_server((interface, port), tls_adapter)
    interface, _host, port = _get_conn_data(tlshttpserver.bind_addr)
    fqdn = interface
    if ip_addr is ANY_INTERFACE_IPV6:
        fqdn = '[{fqdn}]'.format(**locals())
    expect_fallback_response_over_plain_http = adapter_type == 'pyopenssl'
    if expect_fallback_response_over_plain_http:
        resp = requests.get('http://{host!s}:{port!s}/'.format(host=fqdn, port=port), timeout=http_request_timeout)
        assert resp.status_code == 400
        assert resp.text == 'The client sent a plain HTTP request, but this server only speaks HTTPS on this port.'
        return
    with pytest.raises(requests.exceptions.ConnectionError) as ssl_err:
        requests.get('http://{host!s}:{port!s}/'.format(host=fqdn, port=port), timeout=http_request_timeout)
    if IS_LINUX:
        expected_error_code, expected_error_text = (104, 'Connection reset by peer')
    if IS_MACOS:
        expected_error_code, expected_error_text = (54, 'Connection reset by peer')
    if IS_WINDOWS:
        expected_error_code, expected_error_text = (10054, 'An existing connection was forcibly closed by the remote host')
    underlying_error = ssl_err.value.args[0].args[-1]
    err_text = str(underlying_error)
    assert underlying_error.errno == expected_error_code, 'The underlying error is {underlying_error!r}'.format(**locals())
    assert expected_error_text in err_text