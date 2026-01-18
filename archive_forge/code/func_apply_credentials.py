import httplib2
def apply_credentials(credentials, headers):
    if not is_valid(credentials):
        refresh_credentials(credentials)
    return credentials.apply(headers)