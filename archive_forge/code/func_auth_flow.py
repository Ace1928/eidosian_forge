import base64
import httpx
from lazyops.libs import lazyload
from typing import Optional, List, Dict, Any, Union, Iterable, Generator, AsyncGenerator
from .base import BaseModel
from .user_data import AZUserData
from .claims import APIKeyJWTClaims
def auth_flow(self, request: httpx.Request) -> Generator[httpx.Request, httpx.Response, None]:
    """
        Implements the httpx auth flow
        """
    request.headers['Authorization'] = f'Bearer {self.token_flow.token}'
    if self.x_api_key is not None:
        request.headers['x-api-key'] = self.x_api_key
    if self.headers:
        for key, value in self.headers.items():
            if key not in request.headers:
                request.headers[key] = value
    yield request