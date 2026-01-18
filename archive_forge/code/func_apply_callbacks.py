from __future__ import unicode_literals
import re
from tensorboard._vendor import html5lib
from tensorboard._vendor.html5lib.filters.base import Filter
from tensorboard._vendor.html5lib.filters.sanitizer import allowed_protocols
from tensorboard._vendor.html5lib.serializer import HTMLSerializer
from tensorboard._vendor.bleach import callbacks as linkify_callbacks
from tensorboard._vendor.bleach.encoding import force_unicode
from tensorboard._vendor.bleach.utils import alphabetize_attributes
def apply_callbacks(self, attrs, is_new):
    """Given an attrs dict and an is_new bool, runs through callbacks

        Callbacks can return an adjusted attrs dict or ``None``. In the case of
        ``None``, we stop going through callbacks and return that and the link
        gets dropped.

        :arg dict attrs: map of ``(namespace, name)`` -> ``value``

        :arg bool is_new: whether or not this link was added by linkify

        :returns: adjusted attrs dict or ``None``

        """
    for cb in self.callbacks:
        attrs = cb(attrs, is_new)
        if attrs is None:
            return None
    return attrs