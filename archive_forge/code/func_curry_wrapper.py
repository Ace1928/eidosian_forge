from functools import wraps
import hashlib
import json
import os
import pickle
import six.moves.http_client as httplib
from oauth2client import client
from oauth2client import clientsecrets
from oauth2client import transport
from oauth2client.contrib import dictionary_storage
def curry_wrapper(wrapped_function):

    @wraps(wrapped_function)
    def required_wrapper(*args, **kwargs):
        return_url = decorator_kwargs.pop('return_url', request.url)
        requested_scopes = set(self.scopes)
        if scopes is not None:
            requested_scopes |= set(scopes)
        if self.has_credentials():
            requested_scopes |= self.credentials.scopes
        requested_scopes = list(requested_scopes)
        if self.has_credentials() and self.credentials.has_scopes(requested_scopes):
            return wrapped_function(*args, **kwargs)
        else:
            auth_url = self.authorize_url(return_url, scopes=requested_scopes, **decorator_kwargs)
            return redirect(auth_url)
    return required_wrapper