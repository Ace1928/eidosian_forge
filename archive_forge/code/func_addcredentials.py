from suds.transport import *
from suds.transport.http import HttpTransport
import urllib.request, urllib.error, urllib.parse
def addcredentials(self, request):
    credentials = self.credentials()
    if None not in credentials:
        u = credentials[0]
        p = credentials[1]
        self.pm.add_password(None, request.url, u, p)