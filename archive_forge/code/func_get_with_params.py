from django.db import NotSupportedError
from django.db.models import Func, Index
from django.utils.functional import cached_property
def get_with_params(self):
    with_params = []
    if self.fillfactor is not None:
        with_params.append('fillfactor = %d' % self.fillfactor)
    return with_params