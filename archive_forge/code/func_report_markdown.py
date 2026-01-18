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
def report_markdown(self, bundles: dict[str, Any], full_text: bool=True) -> str:
    """create a markdown report"""
    lines = []
    library_names = [len(package.get('name', UNKNOWN_PACKAGE_NAME)) for bundle_name, bundle in bundles.items() for package in bundle.get('packages', [])]
    longest_name = max(library_names) if library_names else 1
    for bundle_name, bundle in bundles.items():
        lines += [f'# {bundle_name}', '']
        packages = bundle.get('packages', [])
        if not packages:
            lines += ['> No licenses found', '']
            continue
        for package in packages:
            name = package.get('name', UNKNOWN_PACKAGE_NAME).strip()
            version_info = package.get('versionInfo', UNKNOWN_PACKAGE_NAME).strip()
            license_id = package.get('licenseId', UNKNOWN_PACKAGE_NAME).strip()
            extracted_text = package.get('extractedText', '')
            lines += ['## ' + '\t'.join([f'**{name}**'.ljust(longest_name), f'`{version_info}`'.ljust(20), license_id])]
            if full_text:
                if not extracted_text:
                    lines += ['', '> No license text available', '']
                else:
                    lines += ['', '', '<pre/>', extracted_text, '</pre>', '']
    return '\n'.join(lines)