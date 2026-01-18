import sys
from os import environ
from os.path import join
from copy import copy
from types import CodeType
from functools import partial
from kivy.factory import Factory
from kivy.lang.parser import (
from kivy.logger import Logger
from kivy.utils import QueryDict
from kivy.cache import Cache
from kivy import kivy_data_dir
from kivy.context import register_context
from kivy.resources import resource_find
from kivy._event import Observable, EventDispatcher
def create_handler(iself, element, key, value, rule, idmap, delayed=False):
    idmap = copy(idmap)
    idmap.update(global_idmap)
    idmap['self'] = iself.proxy_ref
    bound_list = _handlers[iself.uid][key]
    handler_append = bound_list.append
    if delayed:
        fn = delayed_call_fn
        args = [element, key, value, rule, idmap, None]
    else:
        fn = call_fn
        args = (element, key, value, rule, idmap)
    if rule.watched_keys is not None:
        for keys in rule.watched_keys:
            base = idmap.get(keys[0])
            if base is None:
                continue
            f = base = getattr(base, 'proxy_ref', base)
            bound = []
            was_bound = False
            append = bound.append
            k = 1
            for val in keys[1:-1]:
                if isinstance(f, (EventDispatcher, Observable)):
                    prop = f.property(val, True)
                    if prop is not None and getattr(prop, 'rebind', False):
                        uid = f.fbind(val, update_intermediates, base, keys, bound, k, fn, args)
                        append([f.proxy_ref, val, update_intermediates, uid])
                        was_bound = True
                    else:
                        append([f.proxy_ref, val, None, None])
                elif not isinstance(f, type):
                    append([getattr(f, 'proxy_ref', f), val, None, None])
                else:
                    append([f, val, None, None])
                f = getattr(f, val, None)
                if f is None:
                    break
                k += 1
            if isinstance(f, (EventDispatcher, Observable)):
                uid = f.fbind(keys[-1], fn, args)
                if uid:
                    append([f.proxy_ref, keys[-1], fn, uid])
                    was_bound = True
            if was_bound:
                handler_append(bound)
    try:
        return (eval(value, idmap), bound_list)
    except Exception as e:
        tb = sys.exc_info()[2]
        raise BuilderException(rule.ctx, rule.line, '{}: {}'.format(e.__class__.__name__, e), cause=tb)