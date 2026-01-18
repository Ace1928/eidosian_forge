import operator
from functools import reduce
from django.core.exceptions import EmptyResultSet, FullResultSet
from django.db.models.expressions import Case, When
from django.db.models.functions import Mod
from django.db.models.lookups import Exact
from django.utils import tree
from django.utils.functional import cached_property
def replace_expressions(self, replacements):
    if (replacement := replacements.get(self)):
        return replacement
    clone = self.create(connector=self.connector, negated=self.negated)
    for child in self.children:
        clone.children.append(child.replace_expressions(replacements))
    return clone