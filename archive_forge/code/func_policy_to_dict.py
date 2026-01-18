from __future__ import absolute_import, division, print_function
def policy_to_dict(self, rule):
    result = rule.as_dict()
    rights = result['rights']
    if 'Manage' in rights:
        result['rights'] = 'manage'
    elif 'Listen' in rights and 'Send' in rights:
        result['rights'] = 'listen_send'
    else:
        result['rights'] = rights[0].lower()
    return result