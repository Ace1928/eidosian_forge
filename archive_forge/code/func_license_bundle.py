from __future__ import annotations
import asyncio
import csv
import io
import json
import mimetypes
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any
from jupyter_server.base.handlers import APIHandler
from tornado import web
from traitlets import List, Unicode
from traitlets.config import LoggingConfigurable
from .config import get_federated_extensions
def license_bundle(self, path: Path, bundle: str | None) -> dict[str, Any]:
    """Return the content of a packages's license bundles"""
    bundle_json: dict = {'packages': []}
    checked_paths = []
    for license_file in self.third_party_licenses_files:
        licenses_path = path / license_file
        self.log.debug('Loading licenses from %s', licenses_path)
        if not licenses_path.exists():
            checked_paths += [licenses_path]
            continue
        try:
            file_json = json.loads(licenses_path.read_text(encoding='utf-8'))
        except Exception as err:
            self.log.warning('Failed to open third-party licenses for %s: %s\n%s', bundle, licenses_path, err)
            continue
        try:
            bundle_json['packages'].extend(file_json['packages'])
        except Exception as err:
            self.log.warning('Failed to find packages for %s: %s\n%s', bundle, licenses_path, err)
            continue
    if not bundle_json['packages']:
        self.log.warning('Third-party licenses not found for %s: %s', bundle, checked_paths)
    return bundle_json