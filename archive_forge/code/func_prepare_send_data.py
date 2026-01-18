from __future__ import annotations
import dataclasses
from typing import Any, BinaryIO, Dict, Optional, TYPE_CHECKING, Union
import requests
from requests import PreparedRequest
from requests.auth import AuthBase
from requests.structures import CaseInsensitiveDict
from requests_toolbelt.multipart.encoder import MultipartEncoder  # type: ignore
from . import protocol
@staticmethod
def prepare_send_data(files: Optional[Dict[str, Any]]=None, post_data: Optional[Union[Dict[str, Any], bytes, BinaryIO]]=None, raw: bool=False) -> SendData:
    if files:
        if post_data is None:
            post_data = {}
        else:
            if TYPE_CHECKING:
                assert isinstance(post_data, dict)
            for k, v in post_data.items():
                if isinstance(v, bool):
                    v = int(v)
                if isinstance(v, (complex, float, int)):
                    post_data[k] = str(v)
        post_data['file'] = files.get('file')
        post_data['avatar'] = files.get('avatar')
        data = MultipartEncoder(fields=post_data)
        return SendData(data=data, content_type=data.content_type)
    if raw and post_data:
        return SendData(data=post_data, content_type='application/octet-stream')
    if TYPE_CHECKING:
        assert not isinstance(post_data, BinaryIO)
    return SendData(json=post_data, content_type='application/json')