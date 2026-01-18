from pprint import pformat
from six import iteritems
import re
@manual_selector.setter
def manual_selector(self, manual_selector):
    """
        Sets the manual_selector of this V1JobSpec.
        manualSelector controls generation of pod labels and pod selectors.
        Leave `manualSelector` unset unless you are certain what you are doing.
        When false or unset, the system pick labels unique to this job and
        appends those labels to the pod template.  When true, the user is
        responsible for picking unique labels and specifying the selector.
        Failure to pick a unique label may cause this and other jobs to not
        function correctly.  However, You may see `manualSelector=true` in jobs
        that were created with the old `extensions/v1beta1` API. More info:
        https://kubernetes.io/docs/concepts/workloads/controllers/jobs-run-to-completion/#specifying-your-own-pod-selector

        :param manual_selector: The manual_selector of this V1JobSpec.
        :type: bool
        """
    self._manual_selector = manual_selector