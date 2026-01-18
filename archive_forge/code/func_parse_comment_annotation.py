import inspect
from typing import Callable, Optional
from triad.utils.assertion import assert_or_throw
def parse_comment_annotation(func: Callable, annotation: str) -> Optional[str]:
    """Parse comment annotation above the function. It try to find
    comment lines starts with the annotation from bottom up, and will use the first
    occurrance as the result.

    :param func: the function
    :param annotation: the annotation string
    :return: schema hint string

    .. admonition:: Examples

        .. code-block:: python

            # schema: a:int,b:str
            #schema:a:int,b:int # more comment
            # some comment
            def dummy():
                pass

            assert "a:int,b:int" == parse_comment_annotation(dummy, "schema:")
    """
    for orig in reversed((inspect.getcomments(func) or '').splitlines()):
        start = orig.find(':')
        if start <= 0:
            continue
        actual = orig[:start].replace('#', '', 1).strip()
        if actual != annotation:
            continue
        end = orig.find('#', start)
        s = orig[start + 1:end if end > 0 else len(orig)].strip()
        return s
    return None