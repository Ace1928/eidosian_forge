import hmac
import hashlib
import datetime
import requests
def get_aws_request_headers(self, r, aws_access_key, aws_secret_access_key, aws_token):
    """
        Returns a dictionary containing the necessary headers for Amazon's
        signature version 4 signing process. An example return value might
        look like

            {
                'Authorization': 'AWS4-HMAC-SHA256 Credential=YOURKEY/20160618/us-east-1/es/aws4_request, '
                                 'SignedHeaders=host;x-amz-date, '
                                 'Signature=ca0a856286efce2a4bd96a978ca6c8966057e53184776c0685169d08abd74739',
                'x-amz-date': '20160618T220405Z',
            }
        """
    t = datetime.datetime.utcnow()
    amzdate = t.strftime('%Y%m%dT%H%M%SZ')
    datestamp = t.strftime('%Y%m%d')
    canonical_uri = AWSRequestsAuth.get_canonical_path(r)
    canonical_querystring = AWSRequestsAuth.get_canonical_querystring(r)
    canonical_headers = 'host:' + self.aws_host + '\n' + 'x-amz-date:' + amzdate + '\n'
    if aws_token:
        canonical_headers += 'x-amz-security-token:' + aws_token + '\n'
    signed_headers = 'host;x-amz-date'
    if aws_token:
        signed_headers += ';x-amz-security-token'
    body = r.body if r.body else bytes()
    try:
        body = body.encode('utf-8')
    except (AttributeError, UnicodeDecodeError):
        body = body
    payload_hash = hashlib.sha256(body).hexdigest()
    canonical_request = r.method + '\n' + canonical_uri + '\n' + canonical_querystring + '\n' + canonical_headers + '\n' + signed_headers + '\n' + payload_hash
    algorithm = 'AWS4-HMAC-SHA256'
    credential_scope = datestamp + '/' + self.aws_region + '/' + self.service + '/' + 'aws4_request'
    string_to_sign = algorithm + '\n' + amzdate + '\n' + credential_scope + '\n' + hashlib.sha256(canonical_request.encode('utf-8')).hexdigest()
    signing_key = getSignatureKey(aws_secret_access_key, datestamp, self.aws_region, self.service)
    string_to_sign_utf8 = string_to_sign.encode('utf-8')
    signature = hmac.new(signing_key, string_to_sign_utf8, hashlib.sha256).hexdigest()
    authorization_header = algorithm + ' ' + 'Credential=' + aws_access_key + '/' + credential_scope + ', ' + 'SignedHeaders=' + signed_headers + ', ' + 'Signature=' + signature
    headers = {'Authorization': authorization_header, 'x-amz-date': amzdate, 'x-amz-content-sha256': payload_hash}
    if aws_token:
        headers['X-Amz-Security-Token'] = aws_token
    return headers