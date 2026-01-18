from collections import Counter, defaultdict
from functools import partial, reduce
from itertools import chain
from operator import attrgetter, or_
from django.db import IntegrityError, connections, models, transaction
from django.db.models import query_utils, signals, sql
def clear_restricted_objects_from_queryset(self, model, qs):
    if model in self.restricted_objects:
        objs = set(qs.filter(pk__in=[obj.pk for objs in self.restricted_objects[model].values() for obj in objs]))
        self.clear_restricted_objects_from_set(model, objs)