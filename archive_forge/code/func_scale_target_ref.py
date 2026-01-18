from pprint import pformat
from six import iteritems
import re
@scale_target_ref.setter
def scale_target_ref(self, scale_target_ref):
    """
        Sets the scale_target_ref of this V2beta2HorizontalPodAutoscalerSpec.
        scaleTargetRef points to the target resource to scale, and is used to
        the pods for which metrics should be collected, as well as to actually
        change the replica count.

        :param scale_target_ref: The scale_target_ref of this
        V2beta2HorizontalPodAutoscalerSpec.
        :type: V2beta2CrossVersionObjectReference
        """
    if scale_target_ref is None:
        raise ValueError('Invalid value for `scale_target_ref`, must not be `None`')
    self._scale_target_ref = scale_target_ref