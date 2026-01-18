from bisect import bisect_left
from bisect import bisect_right
from contextlib import contextmanager
from copy import deepcopy
from functools import wraps
from inspect import isclass
import calendar
import collections
import datetime
import decimal
import hashlib
import itertools
import logging
import operator
import re
import socket
import struct
import sys
import threading
import time
import uuid
import warnings
def prefetch_add_subquery(sq, subqueries, prefetch_type):
    fixed_queries = [PrefetchQuery(sq)]
    for i, subquery in enumerate(subqueries):
        if isinstance(subquery, tuple):
            subquery, target_model = subquery
        else:
            target_model = None
        if not isinstance(subquery, Query) and is_model(subquery) or isinstance(subquery, ModelAlias):
            subquery = subquery.select()
        subquery_model = subquery.model
        for j in reversed(range(i + 1)):
            fks = backrefs = None
            fixed = fixed_queries[j]
            last_query = fixed.query
            last_model = last_obj = fixed.model
            if isinstance(last_model, ModelAlias):
                last_model = last_model.model
            rels = subquery_model._meta.model_refs.get(last_model, [])
            if rels:
                fks = [getattr(subquery_model, fk.name) for fk in rels]
                pks = [getattr(last_obj, fk.rel_field.name) for fk in rels]
            else:
                backrefs = subquery_model._meta.model_backrefs.get(last_model)
            if (fks or backrefs) and (target_model is last_obj or target_model is None):
                break
        else:
            tgt_err = ' using %s' % target_model if target_model else ''
            raise AttributeError('Error: unable to find foreign key for query: %s%s' % (subquery, tgt_err))
        dest = (target_model,) if target_model else None
        if fks:
            if prefetch_type == PREFETCH_TYPE.WHERE:
                expr = reduce(operator.or_, [fk << last_query.select(pk) for fk, pk in zip(fks, pks)])
                subquery = subquery.where(expr)
            elif prefetch_type == PREFETCH_TYPE.JOIN:
                expr = []
                select_pks = set()
                for fk, pk in zip(fks, pks):
                    expr.append(getattr(last_query.c, pk.column_name) == fk)
                    select_pks.add(pk)
                subquery = subquery.distinct().join(last_query.select(*select_pks), on=reduce(operator.or_, expr))
            fixed_queries.append(PrefetchQuery(subquery, fks, False, dest))
        elif backrefs:
            expr = []
            fields = []
            for backref in backrefs:
                rel_field = getattr(subquery_model, backref.rel_field.name)
                fk_field = getattr(last_obj, backref.name)
                fields.append((rel_field, fk_field))
            if prefetch_type == PREFETCH_TYPE.WHERE:
                for rel_field, fk_field in fields:
                    expr.append(rel_field << last_query.select(fk_field))
                subquery = subquery.where(reduce(operator.or_, expr))
            elif prefetch_type == PREFETCH_TYPE.JOIN:
                select_fks = []
                for rel_field, fk_field in fields:
                    select_fks.append(fk_field)
                    target = getattr(last_query.c, fk_field.column_name)
                    expr.append(rel_field == target)
                subquery = subquery.distinct().join(last_query.select(*select_fks), on=reduce(operator.or_, expr))
            fixed_queries.append(PrefetchQuery(subquery, backrefs, True, dest))
    return fixed_queries