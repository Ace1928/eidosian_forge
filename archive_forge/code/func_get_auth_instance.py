import base64
import datetime
import json
import weakref
import botocore
import botocore.auth
from botocore.awsrequest import create_request_object, prepare_request_dict
from botocore.compat import OrderedDict
from botocore.exceptions import (
from botocore.utils import ArnParser, datetime2timestamp
from botocore.utils import fix_s3_host  # noqa
def get_auth_instance(self, signing_name, region_name, signature_version=None, **kwargs):
    """
        Get an auth instance which can be used to sign a request
        using the given signature version.

        :type signing_name: string
        :param signing_name: Service signing name. This is usually the
                             same as the service name, but can differ. E.g.
                             ``emr`` vs. ``elasticmapreduce``.

        :type region_name: string
        :param region_name: Name of the service region, e.g. ``us-east-1``

        :type signature_version: string
        :param signature_version: Signature name like ``v4``.

        :rtype: :py:class:`~botocore.auth.BaseSigner`
        :return: Auth instance to sign a request.
        """
    if signature_version is None:
        signature_version = self._signature_version
    cls = botocore.auth.AUTH_TYPE_MAPS.get(signature_version)
    if cls is None:
        raise UnknownSignatureVersionError(signature_version=signature_version)
    if cls.REQUIRES_TOKEN is True:
        frozen_token = None
        if self._auth_token is not None:
            frozen_token = self._auth_token.get_frozen_token()
        auth = cls(frozen_token)
        return auth
    credentials = self._credentials
    if getattr(cls, 'REQUIRES_IDENTITY_CACHE', None) is True:
        cache = kwargs['identity_cache']
        key = kwargs['cache_key']
        credentials = cache.get_credentials(key)
        del kwargs['cache_key']
    frozen_credentials = None
    if credentials is not None:
        frozen_credentials = credentials.get_frozen_credentials()
    kwargs['credentials'] = frozen_credentials
    if cls.REQUIRES_REGION:
        if self._region_name is None:
            raise botocore.exceptions.NoRegionError()
        kwargs['region_name'] = region_name
        kwargs['service_name'] = signing_name
    auth = cls(**kwargs)
    return auth