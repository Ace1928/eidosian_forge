from __future__ import unicode_literals
import json
from oauthlib.common import add_params_to_uri, urlencode
def raise_from_error(error, params=None):
    import inspect
    import sys
    kwargs = {'description': params.get('error_description'), 'uri': params.get('error_uri'), 'state': params.get('state')}
    for _, cls in inspect.getmembers(sys.modules[__name__], inspect.isclass):
        if cls.error == error:
            raise cls(**kwargs)
    raise CustomOAuth2Error(error=error, **kwargs)