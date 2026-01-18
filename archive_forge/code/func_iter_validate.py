from __future__ import annotations
import json
import pprint
import warnings
from copy import deepcopy
from pathlib import Path
from textwrap import dedent
from typing import Any, Optional
from ._imports import import_item
from .corpus.words import generate_corpus_id
from .json_compat import ValidationError, _validator_for_name, get_current_validator
from .reader import get_version
from .warnings import DuplicateCellId, MissingIDFieldWarning
def iter_validate(nbdict=None, ref=None, version=None, version_minor=None, relax_add_props=False, nbjson=None, strip_invalid_metadata=False):
    """Checks whether the given notebook dict-like object conforms to the
    relevant notebook format schema.

    Returns a generator of all ValidationErrors if not valid.

    Notes
    -----
    To fix: For security reasons, this function should *never* mutate its `nbdict` argument, and
    should *never* try to validate a mutated or modified version of its notebook.

    """
    if nbdict is not None:
        pass
    elif nbjson is not None:
        nbdict = nbjson
    else:
        msg = "iter_validate() missing 1 required argument: 'nbdict'"
        raise TypeError(msg)
    if version is None:
        version, version_minor = get_version(nbdict)
    if ref:
        try:
            errors = _get_errors(nbdict, version, version_minor, relax_add_props, {'$ref': '#/definitions/%s' % ref})
        except ValidationError as e:
            yield e
            return
    else:
        if strip_invalid_metadata:
            _strip_invalida_metadata(nbdict, version, version_minor, relax_add_props)
        try:
            errors = _get_errors(nbdict, version, version_minor, relax_add_props)
        except ValidationError as e:
            yield e
            return
    for error in errors:
        yield better_validation_error(error, version, version_minor)