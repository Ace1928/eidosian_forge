import logging
import urllib.parse
import smart_open.utils
from ftplib import FTP, FTP_TLS, error_reply
import types
def full_close(self):
    self.orig_close()
    self.socket.close()
    self.conn.close()