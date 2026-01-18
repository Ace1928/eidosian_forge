import uuid
from openstack import utils as sdk_utils
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.common import pagination
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
def print_migrations(self, parsed_args, compute_client, migrations):
    column_headers = ['Source Node', 'Dest Node', 'Source Compute', 'Dest Compute', 'Dest Host', 'Status', 'Server UUID', 'Old Flavor', 'New Flavor', 'Created At', 'Updated At']
    columns = ['source_node', 'dest_node', 'source_compute', 'dest_compute', 'dest_host', 'status', 'server_id', 'old_flavor_id', 'new_flavor_id', 'created_at', 'updated_at']
    if sdk_utils.supports_microversion(compute_client, '2.59'):
        column_headers.insert(0, 'UUID')
        columns.insert(0, 'uuid')
    if sdk_utils.supports_microversion(compute_client, '2.23'):
        column_headers.insert(0, 'Id')
        columns.insert(0, 'id')
        column_headers.insert(len(column_headers) - 2, 'Type')
        columns.insert(len(columns) - 2, 'migration_type')
    if sdk_utils.supports_microversion(compute_client, '2.80'):
        if parsed_args.project:
            column_headers.insert(len(column_headers) - 2, 'Project')
            columns.insert(len(columns) - 2, 'project_id')
        if parsed_args.user:
            column_headers.insert(len(column_headers) - 2, 'User')
            columns.insert(len(columns) - 2, 'user_id')
    return (column_headers, (utils.get_item_properties(mig, columns) for mig in migrations))