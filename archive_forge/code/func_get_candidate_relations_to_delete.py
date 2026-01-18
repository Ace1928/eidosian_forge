from collections import Counter, defaultdict
from functools import partial, reduce
from itertools import chain
from operator import attrgetter, or_
from django.db import IntegrityError, connections, models, transaction
from django.db.models import query_utils, signals, sql
def get_candidate_relations_to_delete(opts):
    return (f for f in opts.get_fields(include_hidden=True) if f.auto_created and (not f.concrete) and (f.one_to_one or f.one_to_many))