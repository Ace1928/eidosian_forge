from django.db.models.lookups import (
def relabeled_clone(self, relabels):
    return self.__class__(relabels.get(self.alias, self.alias), self.targets, self.sources, self.field)