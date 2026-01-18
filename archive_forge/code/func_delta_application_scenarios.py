from ... import errors, osutils, repository, revision, tests, workingtree
from ...tests.scenarios import load_tests_apply_scenarios
from .. import chk_map, groupcompress, inventory
from ..inventory import (ROOT_ID, CHKInventory, DuplicateFileId,
from . import TestCase, TestCaseWithTransport
def delta_application_scenarios():
    scenarios = [('Inventory', {'apply_delta': apply_inventory_Inventory})]
    formats = set()
    for _, format in repository.format_registry.iteritems():
        if format.supports_full_versioned_files:
            scenarios.append((str(format.__name__), {'apply_delta': apply_inventory_Repository_add_inventory_by_delta, 'format': format}))
    for getter in workingtree.format_registry._get_all_lazy():
        try:
            format = getter()
            if callable(format):
                format = format()
        except ImportError:
            pass
        repo_fmt = format._matchingcontroldir.repository_format
        if not repo_fmt.supports_full_versioned_files:
            continue
        scenarios.append((str(format.__class__.__name__) + '.update_basis_by_delta', {'apply_delta': apply_inventory_WT_basis, 'format': format}))
        scenarios.append((str(format.__class__.__name__) + '.apply_inventory_delta', {'apply_delta': apply_inventory_WT, 'format': format}))
    return scenarios