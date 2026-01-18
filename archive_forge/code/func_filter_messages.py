from __future__ import annotations
import abc
import glob
import hashlib
import json
import os
import pathlib
import re
import collections
import collections.abc as c
import typing as t
from ...constants import (
from ...encoding import (
from ...io import (
from ...util import (
from ...util_common import (
from ...ansible_util import (
from ...target import (
from ...executor import (
from ...python_requirements import (
from ...config import (
from ...test import (
from ...data import (
from ...content_config import (
from ...host_configs import (
from ...host_profiles import (
from ...provisioning import (
from ...pypi_proxy import (
from ...venv import (
def filter_messages(self, messages: list[SanityMessage]) -> list[SanityMessage]:
    """Return a filtered list of the given messages using the entries that have been loaded."""
    filtered = []
    for message in messages:
        if message.code in self.test.optional_error_codes and (not self.args.enable_optional_errors):
            continue
        path_entry = self.ignore_entries.get(message.path)
        if path_entry:
            code = message.code if self.code else SanityIgnoreParser.NO_CODE
            line_no = path_entry.get(code)
            if line_no:
                self.used_line_numbers.add(line_no)
                continue
        filtered.append(message)
    return filtered