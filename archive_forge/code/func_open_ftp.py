import urllib.request
import base64
import bisect
import email
import hashlib
import http.client
import io
import os
import posixpath
import re
import socket
import string
import sys
import time
import tempfile
import contextlib
import warnings
from urllib.error import URLError, HTTPError, ContentTooShortError
from urllib.parse import (
from urllib.response import addinfourl, addclosehook
def open_ftp(self, url):
    """Use FTP protocol."""
    if not isinstance(url, str):
        raise URLError('ftp error: proxy support for ftp protocol currently not implemented')
    import mimetypes
    host, path = _splithost(url)
    if not host:
        raise URLError('ftp error: no host given')
    host, port = _splitport(host)
    user, host = _splituser(host)
    if user:
        user, passwd = _splitpasswd(user)
    else:
        passwd = None
    host = unquote(host)
    user = unquote(user or '')
    passwd = unquote(passwd or '')
    host = socket.gethostbyname(host)
    if not port:
        import ftplib
        port = ftplib.FTP_PORT
    else:
        port = int(port)
    path, attrs = _splitattr(path)
    path = unquote(path)
    dirs = path.split('/')
    dirs, file = (dirs[:-1], dirs[-1])
    if dirs and (not dirs[0]):
        dirs = dirs[1:]
    if dirs and (not dirs[0]):
        dirs[0] = '/'
    key = (user, host, port, '/'.join(dirs))
    if len(self.ftpcache) > MAXFTPCACHE:
        for k in list(self.ftpcache):
            if k != key:
                v = self.ftpcache[k]
                del self.ftpcache[k]
                v.close()
    try:
        if key not in self.ftpcache:
            self.ftpcache[key] = ftpwrapper(user, passwd, host, port, dirs)
        if not file:
            type = 'D'
        else:
            type = 'I'
        for attr in attrs:
            attr, value = _splitvalue(attr)
            if attr.lower() == 'type' and value in ('a', 'A', 'i', 'I', 'd', 'D'):
                type = value.upper()
        fp, retrlen = self.ftpcache[key].retrfile(file, type)
        mtype = mimetypes.guess_type('ftp:' + url)[0]
        headers = ''
        if mtype:
            headers += 'Content-Type: %s\n' % mtype
        if retrlen is not None and retrlen >= 0:
            headers += 'Content-Length: %d\n' % retrlen
        headers = email.message_from_string(headers)
        return addinfourl(fp, headers, 'ftp:' + url)
    except ftperrors() as exp:
        raise URLError(f'ftp error: {exp}') from exp