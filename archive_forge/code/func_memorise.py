import builtins
import os
import sys
import types
def memorise(obj, force=False):
    """
    Adds an object to the memo, and recursively adds all the objects
    attributes, and if it is a container, its items. Use force=True to update
    an object already in the memo. Updating is not recursively done.
    """
    obj_id = id(obj)
    if obj_id in memo and (not force) or obj_id in dont_memo:
        return
    id_ = id
    g = get_attrs(obj)
    if g is None:
        attrs_id = None
    else:
        attrs_id = dict(((key, id_(value)) for key, value in g.items()))
    s = get_seq(obj)
    if s is None:
        seq_id = None
    elif hasattr(s, 'items'):
        seq_id = dict(((id_(key), id_(value)) for key, value in s.items()))
    elif not hasattr(s, '__len__'):
        seq_id = None
    else:
        seq_id = [id_(i) for i in s]
    memo[obj_id] = (attrs_id, seq_id)
    id_to_obj[obj_id] = obj
    mem = memorise
    if g is not None:
        [mem(value) for key, value in g.items()]
    if s is not None:
        if hasattr(s, 'items'):
            [(mem(key), mem(item)) for key, item in s.items()]
        elif hasattr(s, '__len__'):
            [mem(item) for item in s]
        else:
            mem(s)