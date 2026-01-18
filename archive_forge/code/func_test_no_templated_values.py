from __future__ import absolute_import, division, print_function
import json
import sys
from awx.main.models import Organization, Team, Project, Inventory
from requests.models import Response
from unittest import mock
def test_no_templated_values(collection_import):
    """This test corresponds to replacements done by
    awx_collection/tools/roles/template_galaxy/tasks/main.yml
    Those replacements should happen at build time, so they should not be
    checked into source.
    """
    ControllerAPIModule = collection_import('plugins.module_utils.controller_api').ControllerAPIModule
    assert ControllerAPIModule._COLLECTION_VERSION == '0.0.1-devel', 'The collection version is templated when the collection is built and the code should retain the placeholder of "0.0.1-devel".'
    InventoryModule = collection_import('plugins.inventory.controller').InventoryModule
    assert InventoryModule.NAME == 'awx.awx.controller', 'The inventory plugin FQCN is templated when the collection is built and the code should retain the default of awx.awx.'