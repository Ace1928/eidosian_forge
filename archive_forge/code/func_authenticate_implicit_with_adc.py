from google.cloud import storage
def authenticate_implicit_with_adc(project_id='your-google-cloud-project-id'):
    """
    When interacting with Google Cloud Client libraries, the library can auto-detect the
    credentials to use.

    // TODO(Developer):
    //  1. Before running this sample,
    //  set up ADC as described in https://cloud.google.com/docs/authentication/external/set-up-adc
    //  2. Replace the project variable.
    //  3. Make sure that the user account or service account that you are using
    //  has the required permissions. For this sample, you must have "storage.buckets.list".
    Args:
        project_id: The project id of your Google Cloud project.
    """
    storage_client = storage.Client(project=project_id)
    buckets = storage_client.list_buckets()
    print('Buckets:')
    for bucket in buckets:
        print(bucket.name)
    print('Listed all storage buckets.')