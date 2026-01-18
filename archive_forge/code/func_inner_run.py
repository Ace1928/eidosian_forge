import errno
import os
import re
import socket
import sys
from datetime import datetime
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from django.core.servers.basehttp import WSGIServer, get_internal_wsgi_application, run
from django.utils import autoreload
from django.utils.regex_helper import _lazy_re_compile
def inner_run(self, *args, **options):
    autoreload.raise_last_exception()
    threading = options['use_threading']
    shutdown_message = options.get('shutdown_message', '')
    if not options['skip_checks']:
        self.stdout.write('Performing system checks...\n\n')
        self.check(display_num_errors=True)
    self.check_migrations()
    try:
        handler = self.get_handler(*args, **options)
        run(self.addr, int(self.port), handler, ipv6=self.use_ipv6, threading=threading, on_bind=self.on_bind, server_cls=self.server_cls)
    except OSError as e:
        ERRORS = {errno.EACCES: "You don't have permission to access that port.", errno.EADDRINUSE: 'That port is already in use.', errno.EADDRNOTAVAIL: "That IP address can't be assigned to."}
        try:
            error_text = ERRORS[e.errno]
        except KeyError:
            error_text = e
        self.stderr.write('Error: %s' % error_text)
        os._exit(1)
    except KeyboardInterrupt:
        if shutdown_message:
            self.stdout.write(shutdown_message)
        sys.exit(0)