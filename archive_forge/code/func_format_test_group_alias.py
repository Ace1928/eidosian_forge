from __future__ import annotations
import dataclasses
import json
import textwrap
import os
import re
import typing as t
from . import (
from ...test import (
from ...config import (
from ...target import (
from ..integration.cloud import (
from ...io import (
from ...util import (
from ...util_common import (
from ...host_configs import (
def format_test_group_alias(self, name: str, fallback: str='') -> str:
    """Return a test group alias using the given name and fallback."""
    group_numbers = self.ci_test_groups.get(name, None)
    if group_numbers:
        if min(group_numbers) != 1:
            display.warning('Min test group "%s" in %s is %d instead of 1.' % (name, self.CI_YML, min(group_numbers)), unique=True)
        if max(group_numbers) != len(group_numbers):
            display.warning('Max test group "%s" in %s is %d instead of %d.' % (name, self.CI_YML, max(group_numbers), len(group_numbers)), unique=True)
        if max(group_numbers) > 9:
            alias = '%s/%s/group(%s)/' % (self.TEST_ALIAS_PREFIX, name, '|'.join((str(i) for i in range(min(group_numbers), max(group_numbers) + 1))))
        elif len(group_numbers) > 1:
            alias = '%s/%s/group[%d-%d]/' % (self.TEST_ALIAS_PREFIX, name, min(group_numbers), max(group_numbers))
        else:
            alias = '%s/%s/group%d/' % (self.TEST_ALIAS_PREFIX, name, min(group_numbers))
    elif fallback:
        alias = '%s/%s/group%d/' % (self.TEST_ALIAS_PREFIX, fallback, 1)
    else:
        raise Exception('cannot find test group "%s" in %s' % (name, self.CI_YML))
    return alias