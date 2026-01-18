import gyp.common
import json
import os
import posixpath
def find_matching_test_target_names(self):
    """Returns the set of output test targets."""
    assert self.is_build_impacted()
    test_target_names_no_all = set(self._test_target_names)
    test_target_names_no_all.discard('all')
    test_targets_no_all = _LookupTargets(test_target_names_no_all, self._unqualified_mapping)
    test_target_names_contains_all = 'all' in self._test_target_names
    if test_target_names_contains_all:
        test_targets = [x for x in set(test_targets_no_all) | set(self._root_targets)]
    else:
        test_targets = [x for x in test_targets_no_all]
    print('supplied test_targets')
    for target_name in self._test_target_names:
        print('\t', target_name)
    print('found test_targets')
    for target in test_targets:
        print('\t', target.name)
    print('searching for matching test targets')
    matching_test_targets = _GetTargetsDependingOnMatchingTargets(test_targets)
    matching_test_targets_contains_all = test_target_names_contains_all and set(matching_test_targets) & set(self._root_targets)
    if matching_test_targets_contains_all:
        matching_test_targets = [x for x in set(matching_test_targets) & set(test_targets_no_all)]
    print('matched test_targets')
    for target in matching_test_targets:
        print('\t', target.name)
    matching_target_names = [gyp.common.ParseQualifiedTarget(target.name)[1] for target in matching_test_targets]
    if matching_test_targets_contains_all:
        matching_target_names.append('all')
        print('\tall')
    return matching_target_names