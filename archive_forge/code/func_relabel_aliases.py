import operator
from functools import reduce
from django.core.exceptions import EmptyResultSet, FullResultSet
from django.db.models.expressions import Case, When
from django.db.models.functions import Mod
from django.db.models.lookups import Exact
from django.utils import tree
from django.utils.functional import cached_property
def relabel_aliases(self, change_map):
    """
        Relabel the alias values of any children. 'change_map' is a dictionary
        mapping old (current) alias values to the new values.
        """
    for pos, child in enumerate(self.children):
        if hasattr(child, 'relabel_aliases'):
            child.relabel_aliases(change_map)
        elif hasattr(child, 'relabeled_clone'):
            self.children[pos] = child.relabeled_clone(change_map)