from pprint import pformat
from six import iteritems
import re
@evaluation_error.setter
def evaluation_error(self, evaluation_error):
    """
        Sets the evaluation_error of this V1beta1SubjectRulesReviewStatus.
        EvaluationError can appear in combination with Rules. It indicates an
        error occurred during rule evaluation, such as an authorizer that
        doesn't support rule evaluation, and that ResourceRules and/or
        NonResourceRules may be incomplete.

        :param evaluation_error: The evaluation_error of this
        V1beta1SubjectRulesReviewStatus.
        :type: str
        """
    self._evaluation_error = evaluation_error