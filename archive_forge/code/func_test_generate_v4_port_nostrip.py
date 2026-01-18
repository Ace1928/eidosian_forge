import testtools
from keystoneclient.contrib.ec2 import utils
from keystoneclient.tests.unit import client_fixtures
def test_generate_v4_port_nostrip(self):
    """Test v4 generator with host:port format for new boto version.

        Validate for new (>=2.9.3) version of boto, where the port should
        not be stripped.
        """
    secret = 'wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY'
    signer = utils.Ec2Signer(secret)
    body_hash = 'b6359072c78d70ebee1e81adcbab4f01bf2c23245fa365ef83fe8f1f955085e2'
    auth_str = 'AWS4-HMAC-SHA256 Credential=AKIAIOSFODNN7EXAMPLE/20110909/us-east-1/iam/aws4_request,SignedHeaders=content-type;host;x-amz-date,'
    headers = {'Content-type': 'application/x-www-form-urlencoded; charset=utf-8', 'X-Amz-Date': '20110909T233600Z', 'Host': 'foo:8000', 'Authorization': auth_str, 'User-Agent': 'Boto/2.9.3 (linux2)'}
    params = {}
    credentials = {'host': 'foo:8000', 'verb': 'POST', 'path': '/', 'params': params, 'headers': headers, 'body_hash': body_hash}
    signature = signer.generate(credentials)
    expected = '26dd92ea79aaa49f533d13b1055acdcd7d7321460d64621f96cc79c4f4d4ab2b'
    self.assertEqual(expected, signature)