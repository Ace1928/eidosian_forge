from itertools import chain, repeat
from textwrap import dedent, indent
from typing import TYPE_CHECKING
from unittest import TestCase
import pytest
from IPython.core.async_helpers import _should_be_async
from IPython.testing.decorators import skip_without
@skip_without('trio')
def test_autoawait_asyncio_wrong_sleep(self):
    iprc('%autoawait asyncio')
    res = iprc_nr('\n        import trio\n        await trio.sleep(0)\n        ')
    with self.assertRaises(RuntimeError):
        res.raise_error()