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
def callbinrepr(op, left: object, right: object) -> Optional[str]:
    """Call the pytest_assertrepr_compare hook and prepare the result.

        This uses the first result from the hook and then ensures the
        following:
        * Overly verbose explanations are truncated unless configured otherwise
          (eg. if running in verbose mode).
        * Embedded newlines are escaped to help util.format_explanation()
          later.
        * If the rewrite mode is used embedded %-characters are replaced
          to protect later % formatting.

        The result can be formatted by util.format_explanation() for
        pretty printing.
        """
    hook_result = ihook.pytest_assertrepr_compare(config=item.config, op=op, left=left, right=right)
    for new_expl in hook_result:
        if new_expl:
            new_expl = truncate.truncate_if_required(new_expl, item)
            new_expl = [line.replace('\n', '\\n') for line in new_expl]
            res = '\n~'.join(new_expl)
            if item.config.getvalue('assertmode') == 'rewrite':
                res = res.replace('%', '%%')
            return res
    return None