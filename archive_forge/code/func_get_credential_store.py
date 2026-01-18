import base64
import json
import logging
import os
import platform
from typing import Any, Dict, Mapping, Optional, Tuple, Union
import dockerpycreds  # type: ignore
def get_credential_store(self, registry: Optional[str]) -> Optional[str]:
    if not registry or registry == INDEX_NAME:
        registry = INDEX_URL
    return self.cred_helpers.get(registry) or self.creds_store