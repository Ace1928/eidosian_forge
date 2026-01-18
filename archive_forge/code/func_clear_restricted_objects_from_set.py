from collections import Counter, defaultdict
from functools import partial, reduce
from itertools import chain
from operator import attrgetter, or_
from django.db import IntegrityError, connections, models, transaction
from django.db.models import query_utils, signals, sql
def clear_restricted_objects_from_set(self, model, objs):
    if model in self.restricted_objects:
        self.restricted_objects[model] = {field: items - objs for field, items in self.restricted_objects[model].items()}