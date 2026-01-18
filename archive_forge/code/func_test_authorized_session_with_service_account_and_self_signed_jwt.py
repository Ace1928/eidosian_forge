import google.auth
import google.auth.credentials
import google.auth.transport.requests
from google.oauth2 import service_account
def test_authorized_session_with_service_account_and_self_signed_jwt():
    credentials, project_id = google.auth.default()
    credentials = credentials.with_scopes(scopes=[], default_scopes=['https://www.googleapis.com/auth/pubsub'])
    http = google.auth.transport.urllib3.AuthorizedHttp(credentials=credentials, default_host='pubsub.googleapis.com')
    response = http.urlopen(method='GET', url='https://pubsub.googleapis.com/v1/projects/{}/topics'.format(project_id))
    assert response.status == 200
    assert credentials._jwt_credentials is not None
    assert credentials._jwt_credentials.token == credentials.token