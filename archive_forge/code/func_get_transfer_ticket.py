import logging
import posixpath
import random
import re
import http.client as httplib
import urllib.parse as urlparse
from oslo_vmware._i18n import _
from oslo_vmware import constants
from oslo_vmware import exceptions
from oslo_vmware import vim_util
def get_transfer_ticket(self, session, method):
    client_factory = session.vim.client.factory
    spec = vim_util.get_http_service_request_spec(client_factory, method, str(self))
    ticket = session.invoke_api(session.vim, 'AcquireGenericServiceTicket', session.vim.service_content.sessionManager, spec=spec)
    return '%s="%s"' % (constants.CGI_COOKIE_KEY, ticket.id)