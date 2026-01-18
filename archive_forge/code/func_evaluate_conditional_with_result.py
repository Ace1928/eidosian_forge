from __future__ import (absolute_import, division, print_function)
import typing as t
from ansible.errors import AnsibleError, AnsibleUndefinedVariable, AnsibleTemplateError
from ansible.module_utils.common.text.converters import to_native
from ansible.playbook.attribute import FieldAttribute
from ansible.template import Templar
from ansible.utils.display import Display
def evaluate_conditional_with_result(self, templar: Templar, all_vars: dict[str, t.Any]) -> tuple[bool, t.Optional[str]]:
    """Loops through the conditionals set on this object, returning
        False if any of them evaluate as such as well as the condition
        that was false.
        """
    for conditional in self.when:
        if conditional is None or conditional == '':
            res = True
        elif isinstance(conditional, bool):
            res = conditional
        else:
            try:
                res = self._check_conditional(conditional, templar, all_vars)
            except AnsibleError as e:
                raise AnsibleError("The conditional check '%s' failed. The error was: %s" % (to_native(conditional), to_native(e)), obj=getattr(self, '_ds', None))
        display.debug('Evaluated conditional (%s): %s' % (conditional, res))
        if not res:
            return (res, conditional)
    return (True, None)