from novaclient.tests.unit import fakes
from novaclient.tests.unit.fixture_data import base
def post_images_1_metadata(request, context):
    body = request.json()
    assert list(body) == ['metadata']
    fakes.assert_has_keys(body['metadata'], required=['test_key'])
    return {'metadata': image_1['metadata']}