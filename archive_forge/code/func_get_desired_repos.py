from __future__ import absolute_import, division, print_function
from ansible_collections.theforeman.foreman.plugins.module_utils.foreman_helper import KatelloEntityAnsibleModule
def get_desired_repos(desired_substitutions, available_repos):
    desired_repos = []
    for sub in desired_substitutions:
        desired_repos += filter(lambda available: available['substitutions'] == sub, available_repos)
    return desired_repos