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
def authorize_view(self):
    """Flask view that starts the authorization flow.

        Starts flow by redirecting the user to the OAuth2 provider.
        """
    args = request.args.to_dict()
    args['scopes'] = request.args.getlist('scopes')
    return_url = args.pop('return_url', None)
    if return_url is None:
        return_url = request.referrer or '/'
    flow = self._make_flow(return_url=return_url, **args)
    auth_url = flow.step1_get_authorize_url()
    return redirect(auth_url)