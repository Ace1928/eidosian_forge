from __future__ import absolute_import
import itertools
from time import time
from . import Errors
from . import DebugFlags
from . import Options
from .Errors import CompileError, InternalError, AbortError
from . import Naming
def insert_into_pipeline(pipeline, transform, before=None, after=None):
    """
    Insert a new transform into the pipeline after or before an instance of
    the given class. e.g.

        pipeline = insert_into_pipeline(pipeline, transform,
                                        after=AnalyseDeclarationsTransform)
    """
    assert before or after
    cls = before or after
    for i, t in enumerate(pipeline):
        if isinstance(t, cls):
            break
    if after:
        i += 1
    return pipeline[:i] + [transform] + pipeline[i:]