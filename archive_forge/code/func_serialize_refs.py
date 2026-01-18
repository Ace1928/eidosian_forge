import os
import warnings
from contextlib import suppress
from typing import Any, Dict, Optional, Set
from .errors import PackedRefsException, RefFormatError
from .file import GitFile, ensure_dir_exists
from .objects import ZERO_SHA, ObjectID, Tag, git_line, valid_hexsha
from .pack import ObjectContainer
def serialize_refs(store, refs):
    from .object_store import peel_sha
    ret = {}
    for ref, sha in refs.items():
        try:
            unpeeled, peeled = peel_sha(store, sha)
        except KeyError:
            warnings.warn('ref {} points at non-present sha {}'.format(ref.decode('utf-8', 'replace'), sha.decode('ascii')), UserWarning)
            continue
        else:
            if isinstance(unpeeled, Tag):
                ret[ref + PEELED_TAG_SUFFIX] = peeled.id
            ret[ref] = unpeeled.id
    return ret