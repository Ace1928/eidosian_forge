import logging
from botocore import xform_name
@property
def batch_actions(self):
    """
        Get a list of batch actions for this resource.

        :type: list(:py:class:`Action`)
        """
    actions = []
    for name, item in self._definition.get('batchActions', {}).items():
        name = self._get_name('batch_action', name)
        actions.append(Action(name, item, self._resource_defs))
    return actions