import httplib2
def get_credentials_from_http(http):
    if http is None:
        return None
    elif hasattr(http.request, 'credentials'):
        return http.request.credentials
    elif hasattr(http, 'credentials') and (not isinstance(http.credentials, httplib2.Credentials)):
        return http.credentials
    else:
        return None