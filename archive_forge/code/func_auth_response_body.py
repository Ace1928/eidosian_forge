import uuid
from keystoneauth1 import fixture
def auth_response_body():
    f = fixture.V2Token(token_id='ab48a9efdfedb23ty3494', expires='2010-11-01T03:32:15-05:00', tenant_id='345', tenant_name='My Project', user_id='123', user_name='jqsmith', audit_chain_id=uuid.uuid4().hex)
    f.add_role(id='234', name='compute:admin')
    role = f.add_role(id='235', name='object-store:admin')
    role['tenantId'] = '1'
    s = f.add_service('compute', 'Cloud Servers')
    endpoint = s.add_endpoint(public='https://compute.north.host/v1/1234', internal='https://compute.north.host/v1/1234', region='North')
    endpoint['tenantId'] = '1'
    endpoint['versionId'] = '1.0'
    endpoint['versionInfo'] = 'https://compute.north.host/v1.0/'
    endpoint['versionList'] = 'https://compute.north.host/'
    endpoint = s.add_endpoint(public='https://compute.north.host/v1.1/3456', internal='https://compute.north.host/v1.1/3456', region='North')
    endpoint['tenantId'] = '2'
    endpoint['versionId'] = '1.1'
    endpoint['versionInfo'] = 'https://compute.north.host/v1.1/'
    endpoint['versionList'] = 'https://compute.north.host/'
    s = f.add_service('object-store', 'Cloud Files')
    endpoint = s.add_endpoint(public='https://swift.north.host/v1/blah', internal='https://swift.north.host/v1/blah', region='South')
    endpoint['tenantId'] = '11'
    endpoint['versionId'] = '1.0'
    endpoint['versionInfo'] = 'uri'
    endpoint['versionList'] = 'uri'
    endpoint = s.add_endpoint(public='https://swift.north.host/v1.1/blah', internal='https://compute.north.host/v1.1/blah', region='South')
    endpoint['tenantId'] = '2'
    endpoint['versionId'] = '1.1'
    endpoint['versionInfo'] = 'https://swift.north.host/v1.1/'
    endpoint['versionList'] = 'https://swift.north.host/'
    s = f.add_service('image', 'Image Servers')
    s.add_endpoint(public='https://image.north.host/v1/', internal='https://image-internal.north.host/v1/', region='North')
    s.add_endpoint(public='https://image.south.host/v1/', internal='https://image-internal.south.host/v1/', region='South')
    return f