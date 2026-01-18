from itertools import chain
from django.utils.inspect import func_accepts_kwargs
from django.utils.itercompat import is_iterable
def tag_exists(self, tag, include_deployment_checks=False):
    return tag in self.tags_available(include_deployment_checks)