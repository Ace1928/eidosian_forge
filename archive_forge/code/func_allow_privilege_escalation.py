from pprint import pformat
from six import iteritems
import re
@allow_privilege_escalation.setter
def allow_privilege_escalation(self, allow_privilege_escalation):
    """
        Sets the allow_privilege_escalation of this V1SecurityContext.
        AllowPrivilegeEscalation controls whether a process can gain more
        privileges than its parent process. This bool directly controls if the
        no_new_privs flag will be set on the container process.
        AllowPrivilegeEscalation is true always when the container is: 1) run as
        Privileged 2) has CAP_SYS_ADMIN

        :param allow_privilege_escalation: The allow_privilege_escalation of
        this V1SecurityContext.
        :type: bool
        """
    self._allow_privilege_escalation = allow_privilege_escalation