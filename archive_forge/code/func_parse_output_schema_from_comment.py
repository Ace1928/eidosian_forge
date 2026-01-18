import inspect
from typing import Callable, Optional
from triad.utils.assertion import assert_or_throw
def parse_output_schema_from_comment(func: Callable) -> Optional[str]:
    """Parse schema hint from the comments above the function. It try to find
    comment lines starts with `schema:` from bottom up, and will use the first
    occurrance as the hint.

    :param func: the function
    :return: schema hint string

    .. admonition:: Examples

        .. code-block:: python

            # schema: a:int,b:str
            #schema:a:int,b:int # more comment
            # some comment
            def dummy():
                pass

            assert "a:int,b:int" == parse_output_schema_from_comment(dummy)
    """
    res = parse_comment_annotation(func, _COMMENT_SCHEMA_ANNOTATION)
    if res is None:
        return None
    assert_or_throw(res != '', SyntaxError('incorrect schema annotation'))
    return res.strip()