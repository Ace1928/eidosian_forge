import httplib2
def refresh_credentials(credentials):
    refresh_http = httplib2.Http()
    if HAS_GOOGLE_AUTH and isinstance(credentials, google.auth.credentials.Credentials):
        request = google_auth_httplib2.Request(refresh_http)
        return credentials.refresh(request)
    else:
        return credentials.refresh(refresh_http)