from pprint import pformat
from six import iteritems
import re
@template_generation.setter
def template_generation(self, template_generation):
    """
        Sets the template_generation of this V1beta1DaemonSetSpec.
        DEPRECATED. A sequence number representing a specific generation of the
        template. Populated by the system. It can be set only during the
        creation.

        :param template_generation: The template_generation of this
        V1beta1DaemonSetSpec.
        :type: int
        """
    self._template_generation = template_generation