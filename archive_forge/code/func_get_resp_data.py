from ._base import *
from .models import component_name, rename_if_scope_child_component
from fastapi import status
def get_resp_data(self):
    return self.raw_data