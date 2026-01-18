import abc
import base64
import time
from abc import ABC
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Dict, Optional, Union
import jwt
from requests import utils
from github import Consts
from github.InstallationAuthorization import InstallationAuthorization
from github.Requester import Requester, WithRequester
def create_jwt(self, expiration: Optional[int]=None) -> str:
    """
        Create a signed JWT
        https://docs.github.com/en/developers/apps/building-github-apps/authenticating-with-github-apps#authenticating-as-a-github-app

        :return string: jwt
        """
    if expiration is not None:
        assert isinstance(expiration, int), expiration
        assert Consts.MIN_JWT_EXPIRY <= expiration <= Consts.MAX_JWT_EXPIRY, expiration
    now = int(time.time())
    payload = {'iat': now + self._jwt_issued_at, 'exp': now + (expiration if expiration is not None else self._jwt_expiry), 'iss': self._app_id}
    encrypted = jwt.encode(payload, key=self.private_key, algorithm=self._jwt_algorithm)
    if isinstance(encrypted, bytes):
        return encrypted.decode('utf-8')
    return encrypted