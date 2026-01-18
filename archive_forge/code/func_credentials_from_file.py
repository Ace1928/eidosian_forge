import httplib2
def credentials_from_file(filename, scopes=None, quota_project_id=None):
    """Returns credentials loaded from a file."""
    if HAS_GOOGLE_AUTH:
        credentials, _ = google.auth.load_credentials_from_file(filename, scopes=scopes, quota_project_id=quota_project_id)
        return credentials
    else:
        raise EnvironmentError('client_options.credentials_file is only supported in google-auth.')