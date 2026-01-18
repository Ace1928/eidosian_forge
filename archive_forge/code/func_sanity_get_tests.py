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
@cache
def sanity_get_tests() -> tuple[SanityTest, ...]:
    """Return a tuple of the available sanity tests."""
    import_plugins('commands/sanity')
    sanity_plugins: dict[str, t.Type[SanityTest]] = {}
    load_plugins(SanityTest, sanity_plugins)
    sanity_plugins.pop('sanity')
    sanity_tests = tuple((plugin() for plugin in sanity_plugins.values() if data_context().content.is_ansible or not plugin.ansible_only))
    sanity_tests = tuple(sorted(sanity_tests + collect_code_smell_tests(), key=lambda k: k.name))
    return sanity_tests