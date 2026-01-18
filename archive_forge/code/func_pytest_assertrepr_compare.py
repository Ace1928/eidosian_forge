import sys
from typing import Any
from typing import Generator
from typing import List
from typing import Optional
from typing import TYPE_CHECKING
from _pytest.assertion import rewrite
from _pytest.assertion import truncate
from _pytest.assertion import util
from _pytest.assertion.rewrite import assertstate_key
from _pytest.config import Config
from _pytest.config import hookimpl
from _pytest.config.argparsing import Parser
from _pytest.nodes import Item
def pytest_assertrepr_compare(config: Config, op: str, left: Any, right: Any) -> Optional[List[str]]:
    return util.assertrepr_compare(config=config, op=op, left=left, right=right)