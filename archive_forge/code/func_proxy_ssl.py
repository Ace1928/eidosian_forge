from datetime import datetime
import errno
import os
import random
import re
import socket
import sys
import time
import xml.sax
import copy
from boto import auth
from boto import auth_handler
import boto
import boto.utils
import boto.handler
import boto.cacerts
from boto import config, UserAgent
from boto.compat import six, http_client, urlparse, quote, encodebytes
from boto.exception import AWSConnectionError
from boto.exception import BotoClientError
from boto.exception import BotoServerError
from boto.exception import PleaseRetryException
from boto.exception import S3ResponseError
from boto.provider import Provider
from boto.resultset import ResultSet
def proxy_ssl(self, host=None, port=None):
    if host and port:
        host = '%s:%d' % (host, port)
    else:
        host = '%s:%d' % (self.host, self.port)
    timeout = self.http_connection_kwargs.get('timeout')
    if timeout is not None:
        sock = socket.create_connection((self.proxy, int(self.proxy_port)), timeout)
    else:
        sock = socket.create_connection((self.proxy, int(self.proxy_port)))
    boto.log.debug('Proxy connection: CONNECT %s HTTP/1.0\r\n', host)
    sock.sendall(six.ensure_binary('CONNECT %s HTTP/1.0\r\n' % host))
    sock.sendall(six.ensure_binary('User-Agent: %s\r\n' % UserAgent))
    if self.proxy_user and self.proxy_pass:
        for k, v in self.get_proxy_auth_header().items():
            sock.sendall(six.ensure_binary('%s: %s\r\n' % (k, v)))
        if config.getbool('Boto', 'send_crlf_after_proxy_auth_headers', False):
            sock.sendall(six.ensure_binary('\r\n'))
    else:
        sock.sendall(six.ensure_binary('\r\n'))
    resp = http_client.HTTPResponse(sock, debuglevel=self.debug)
    resp.begin()
    if resp.status != 200:
        raise socket.error(-71, six.ensure_binary('Error talking to HTTP proxy %s:%s: %s (%s)' % (self.proxy, self.proxy_port, resp.status, resp.reason)))
    resp.close()
    h = http_client.HTTPConnection(host)
    if self.https_validate_certificates and HAVE_HTTPS_CONNECTION:
        msg = 'wrapping ssl socket for proxied connection; '
        if self.ca_certificates_file:
            msg += 'CA certificate file=%s' % self.ca_certificates_file
        else:
            msg += 'using system provided SSL certs'
        boto.log.debug(msg)
        key_file = self.http_connection_kwargs.get('key_file', None)
        cert_file = self.http_connection_kwargs.get('cert_file', None)
        context = ssl.create_default_context()
        context.verify_mode = ssl.CERT_REQUIRED
        context.check_hostname = True
        if cert_file:
            context.load_cert_chain(cert_file, key_file)
        context.load_verify_locations(self.ca_certificates_file)
        sslSock = context.wrap_socket(sock, server_hostname=self.host)
        cert = sslSock.getpeercert()
        hostname = self.host.split(':', 0)[0]
        if not https_connection.ValidateCertificateHostname(cert, hostname):
            raise https_connection.InvalidCertificateException(hostname, cert, 'hostname mismatch')
    elif hasattr(http_client, 'ssl'):
        sslSock = http_client.ssl.SSLSocket(sock)
    else:
        sslSock = socket.ssl(sock, None, None)
        sslSock = http_client.FakeSocket(sock, sslSock)
    h.sock = sslSock
    return h