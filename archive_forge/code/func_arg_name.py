from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import six
@property
def arg_name(self):
    """A string property representing the final argument name."""
    return self.concept.GetPresentationName()