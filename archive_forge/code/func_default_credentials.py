import httplib2
def default_credentials(scopes=None, quota_project_id=None):
    """Returns Application Default Credentials."""
    if HAS_GOOGLE_AUTH:
        credentials, _ = google.auth.default(scopes=scopes, quota_project_id=quota_project_id)
        return credentials
    elif HAS_OAUTH2CLIENT:
        if scopes is not None or quota_project_id is not None:
            raise EnvironmentError('client_options.scopes and client_options.quota_project_id are not supported in oauth2client.Please install google-auth.')
        return oauth2client.client.GoogleCredentials.get_application_default()
    else:
        raise EnvironmentError('No authentication library is available. Please install either google-auth or oauth2client.')