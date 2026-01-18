from google.cloud import storage
import google.oauth2.credentials
import google.auth
def authenticate_explicit_with_adc():
    """
    List storage buckets by authenticating with ADC.

    // TODO(Developer):
    //  1. Before running this sample,
    //  set up ADC as described in https://cloud.google.com/docs/authentication/external/set-up-adc
    //  2. Replace the project variable.
    //  3. Make sure you have the necessary permission to list storage buckets: "storage.buckets.list"
    """
    credentials, project_id = google.auth.default()
    storage_client = storage.Client(credentials=credentials, project=project_id)
    buckets = storage_client.list_buckets()
    print('Buckets:')
    for bucket in buckets:
        print(bucket.name)
    print('Listed all storage buckets.')