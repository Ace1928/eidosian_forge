import copy
import json
import urllib.parse
import requests
@property
def allow_redirects(self):
    return self._allow_redirects