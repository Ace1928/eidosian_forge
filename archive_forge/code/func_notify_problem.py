import atexit
import traceback
import io
import socket, sys, threading
import posixpath
import time
import os
from itertools import count
import _thread
import queue
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from urllib.parse import unquote, urlsplit
from paste.util import converters
import logging
def notify_problem(self, msg, subject=None, spawn_thread=True):
    """
        Called when there's a substantial problem.  msg contains the
        body of the notification, subject the summary.

        If spawn_thread is true, then the email will be send in
        another thread (so this doesn't block).
        """
    if not self.error_email:
        return
    if spawn_thread:
        t = threading.Thread(target=self.notify_problem, args=(msg, subject, False))
        t.start()
        return
    from_address = 'errors@localhost'
    if not subject:
        subject = msg.strip().splitlines()[0]
        subject = subject[:50]
        subject = '[http threadpool] %s' % subject
    headers = ['To: %s' % self.error_email, 'From: %s' % from_address, 'Subject: %s' % subject]
    try:
        system = ' '.join(os.uname())
    except:
        system = '(unknown)'
    body = 'An error has occurred in the paste.httpserver.ThreadPool\nError:\n  %(msg)s\nOccurred at: %(time)s\nPID: %(pid)s\nSystem: %(system)s\nServer .py file: %(file)s\n' % dict(msg=msg, time=time.strftime('%c'), pid=os.getpid(), system=system, file=os.path.abspath(__file__))
    message = '\n'.join(headers) + '\n\n' + body
    import smtplib
    server = smtplib.SMTP('localhost')
    error_emails = [e.strip() for e in self.error_email.split(',') if e.strip()]
    server.sendmail(from_address, error_emails, message)
    server.quit()
    print('email sent to', error_emails, message)