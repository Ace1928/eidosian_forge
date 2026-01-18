from ._base import *
from .models import component_name, rename_if_scope_child_component
from fastapi import status
def get_resp(self):
    error = {'code': self.CODE, 'message': self.MESSAGE}
    resp_data = self.get_resp_data()
    if resp_data:
        error['data'] = resp_data
    resp = {'jsonrpc': '2.0', 'error': error, 'id': None}
    return jsonable_encoder(resp)