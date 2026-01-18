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
def retrfile(self, file, type):
    import ftplib
    self.endtransfer()
    if type in ('d', 'D'):
        cmd = 'TYPE A'
        isdir = 1
    else:
        cmd = 'TYPE ' + type
        isdir = 0
    try:
        self.ftp.voidcmd(cmd)
    except ftplib.all_errors:
        self.init()
        self.ftp.voidcmd(cmd)
    conn = None
    if file and (not isdir):
        try:
            cmd = 'RETR ' + file
            conn, retrlen = self.ftp.ntransfercmd(cmd)
        except ftplib.error_perm as reason:
            if str(reason)[:3] != '550':
                raise URLError(f'ftp error: {reason}') from reason
    if not conn:
        self.ftp.voidcmd('TYPE A')
        if file:
            pwd = self.ftp.pwd()
            try:
                try:
                    self.ftp.cwd(file)
                except ftplib.error_perm as reason:
                    raise URLError('ftp error: %r' % reason) from reason
            finally:
                self.ftp.cwd(pwd)
            cmd = 'LIST ' + file
        else:
            cmd = 'LIST'
        conn, retrlen = self.ftp.ntransfercmd(cmd)
    self.busy = 1
    ftpobj = addclosehook(conn.makefile('rb'), self.file_close)
    self.refcount += 1
    conn.close()
    return (ftpobj, retrlen)