from __future__ import annotations
import logging
import os
from typing import TYPE_CHECKING, List, Optional, Tuple
def build_resource_service(credentials: Optional[Credentials]=None, service_name: str='gmail', service_version: str='v1') -> Resource:
    """Build a Gmail service."""
    credentials = credentials or get_gmail_credentials()
    builder = import_googleapiclient_resource_builder()
    return builder(service_name, service_version, credentials=credentials)