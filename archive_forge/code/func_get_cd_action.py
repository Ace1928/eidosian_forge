from __future__ import (absolute_import, division, print_function)
def get_cd_action(self, current, desired):
    """ takes a desired state and a current state, and return an action:
            create, delete, None
            eg:
            is_present = 'absent'
            some_object = self.get_object(source)
            if some_object is not None:
                is_present = 'present'
            action = cd_action(current=is_present, desired = self.desired.state())
        """
    if 'state' in desired:
        desired_state = desired['state']
    else:
        desired_state = 'present'
    if current is None and desired_state == 'absent':
        return None
    if current is not None and desired_state == 'present':
        return None
    self.changed = True
    if current is not None:
        return 'delete'
    return 'create'