import datetime
import json
import os
import socket
from oauth2client import _helpers
from oauth2client import client
@property
def project_id(self):
    return self.devshell_response.project_id