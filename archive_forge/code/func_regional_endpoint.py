from google.cloud.pubsublite.types import CloudRegion
def regional_endpoint(region: CloudRegion):
    return f'dns:///{region}-pubsublite.googleapis.com'