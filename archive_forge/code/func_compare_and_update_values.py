from __future__ import (absolute_import, division, print_function)
def compare_and_update_values(self, current, desired, keys_to_compare):
    updated_values = dict()
    is_changed = False
    for key in keys_to_compare:
        if key in current:
            if key in desired and desired[key] is not None:
                if current[key] != desired[key]:
                    updated_values[key] = desired[key]
                    is_changed = True
                else:
                    updated_values[key] = current[key]
            else:
                updated_values[key] = current[key]
    return (updated_values, is_changed)