import difflib
import inspect
import pickle
import traceback
from collections import defaultdict
from contextlib import contextmanager
import numpy as np
import param
from .accessors import Opts  # noqa (clean up in 2.0)
from .pprint import InfoPrinter
from .tree import AttrTree
from .util import group_sanitizer, label_sanitizer, sanitize_identifier
@classmethod
def create_custom_trees(cls, obj, options=None, backend=None):
    """
        Returns the appropriate set of customized subtree clones for
        an object, suitable for merging with Store.custom_options (i.e
        with the ids appropriately offset). Note if an object has no
        integer ids a new OptionTree is built.

        The id_mapping return value is a list mapping the ids that
        need to be matched as set to their new values.
        """
    clones, id_mapping = ({}, [])
    obj_ids = cls.get_object_ids(obj)
    offset = cls.id_offset()
    obj_ids = [None] if len(obj_ids) == 0 else obj_ids
    used_obj_types = [(opt.split('.')[0],) for opt in options]
    backend = backend or Store.current_backend
    available_options = Store.options(backend=backend)
    used_options = {}
    for obj_type in available_options:
        if obj_type in used_obj_types:
            opts_groups = available_options[obj_type].groups
            used_options[obj_type] = {grp: Options(allowed_keywords=opt.allowed_keywords, backend=backend) for grp, opt in opts_groups.items()}
    custom_options = Store.custom_options(backend=backend)
    for tree_id in obj_ids:
        if tree_id is not None and tree_id in custom_options:
            original = custom_options[tree_id]
            clone = OptionTree(items=original.items(), groups=original.groups, backend=original.backend)
            clones[tree_id + offset + 1] = clone
            id_mapping.append((tree_id, tree_id + offset + 1))
        else:
            clone = OptionTree(groups=available_options.groups, backend=backend)
            clones[offset] = clone
            id_mapping.append((tree_id, offset))
        for obj_type, opts in used_options.items():
            clone[obj_type] = opts
    return ({k: cls.apply_customizations(options, t) if options else t for k, t in clones.items()}, id_mapping)