import urllib
import uuid
from boto.connection import AWSQueryConnection
from boto.fps.exception import ResponseErrorFactory
from boto.fps.response import ResponseFactory
import boto.fps.response
@needs_caller_reference
@requires(['returnURL', 'pipelineName'])
def cbui_url(self, **kw):
    """
        Generate a signed URL for the Co-Branded service API given arguments as
        payload.
        """
    sandbox = 'sandbox' in self.host and 'payments-sandbox' or 'payments'
    endpoint = 'authorize.{0}.amazon.com'.format(sandbox)
    base = '/cobranded-ui/actions/start'
    validpipelines = ('SingleUse', 'MultiUse', 'Recurring', 'Recipient', 'SetupPrepaid', 'SetupPostpaid', 'EditToken')
    assert kw['pipelineName'] in validpipelines, 'Invalid pipelineName'
    kw.update({'signatureMethod': 'HmacSHA256', 'signatureVersion': '2'})
    kw.setdefault('callerKey', self.aws_access_key_id)
    safestr = lambda x: x is not None and str(x) or ''
    safequote = lambda x: urllib.quote(safestr(x), safe='~')
    payload = sorted([(k, safequote(v)) for k, v in kw.items()])
    encoded = lambda p: '&'.join([k + '=' + v for k, v in p])
    canonical = '\n'.join(['GET', endpoint, base, encoded(payload)])
    signature = self._auth_handler.sign_string(canonical)
    payload += [('signature', safequote(signature))]
    payload.sort()
    return 'https://{0}{1}?{2}'.format(endpoint, base, encoded(payload))