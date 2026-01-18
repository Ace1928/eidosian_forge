from pprint import pformat
from six import iteritems
import re
@resource_rules.setter
def resource_rules(self, resource_rules):
    """
        Sets the resource_rules of this V1beta1SubjectRulesReviewStatus.
        ResourceRules is the list of actions the subject is allowed to perform
        on resources. The list ordering isn't significant, may contain
        duplicates, and possibly be incomplete.

        :param resource_rules: The resource_rules of this
        V1beta1SubjectRulesReviewStatus.
        :type: list[V1beta1ResourceRule]
        """
    if resource_rules is None:
        raise ValueError('Invalid value for `resource_rules`, must not be `None`')
    self._resource_rules = resource_rules