from itertools import chain, repeat
from textwrap import dedent, indent
from typing import TYPE_CHECKING
from unittest import TestCase
import pytest
from IPython.core.async_helpers import _should_be_async
from IPython.testing.decorators import skip_without
@skip_without('trio')
def test_autoawait_trio(self):
    iprc('%autoawait trio')