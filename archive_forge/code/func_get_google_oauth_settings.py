import base64
import binascii
import hashlib
import hmac
import time
import urllib.parse
import uuid
import warnings
from tornado import httpclient
from tornado import escape
from tornado.httputil import url_concat
from tornado.util import unicode_type
from tornado.web import RequestHandler
from typing import List, Any, Dict, cast, Iterable, Union, Optional
def get_google_oauth_settings(self) -> Dict[str, str]:
    """Return the Google OAuth 2.0 credentials that you created with
        [Google Cloud
        Platform](https://console.cloud.google.com/apis/credentials). The dict
        format is::

            {
                "key": "your_client_id", "secret": "your_client_secret"
            }

        If your credentials are stored differently (e.g. in a db) you can
        override this method for custom provision.
        """
    handler = cast(RequestHandler, self)
    return handler.settings[self._OAUTH_SETTINGS_KEY]