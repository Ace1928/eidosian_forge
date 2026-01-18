from __future__ import (absolute_import, division, print_function)
def ingate_create_client(**kwargs):
    api_client = ingate_create_client_noauth(**kwargs)
    api_client.authenticate()
    return api_client