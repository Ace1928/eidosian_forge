import base64
import json
import logging
import os
import platform
from typing import Any, Dict, Mapping, Optional, Tuple, Union
import dockerpycreds  # type: ignore
def get_all_credentials(self) -> Dict[str, Dict[str, Any]]:
    auth_data = self.auths.copy()
    if self.creds_store:
        store = self._get_store_instance(self.creds_store)
        for k in store.list().keys():
            auth_data[k] = self._resolve_authconfig_credstore(k, self.creds_store)
    for reg, store_name in self.cred_helpers.items():
        auth_data[reg] = self._resolve_authconfig_credstore(reg, store_name)
    return auth_data