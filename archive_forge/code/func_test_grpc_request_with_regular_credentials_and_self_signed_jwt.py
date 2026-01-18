import google.auth
import google.auth.credentials
import google.auth.jwt
import google.auth.transport.grpc
from google.oauth2 import service_account
from google.cloud import pubsub_v1
def test_grpc_request_with_regular_credentials_and_self_signed_jwt(http_request):
    credentials, project_id = google.auth.default()
    credentials = credentials.with_scopes(scopes=[], default_scopes=['https://www.googleapis.com/auth/pubsub'])
    credentials._create_self_signed_jwt(audience='https://pubsub.googleapis.com/')
    client = pubsub_v1.PublisherClient(credentials=credentials)
    list_topics_iter = client.list_topics(project='projects/{}'.format(project_id))
    list(list_topics_iter)
    assert credentials._jwt_credentials is not None
    assert credentials._jwt_credentials.token == credentials.token