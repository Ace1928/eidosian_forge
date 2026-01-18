from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.command_lib.transfer.appliances import flags
def apply_args_to_appliance(appliance_resource, args):
    """Maps command arguments to appliance resource values.

  Args:
    appliance_resource (messages.Appliance): The target appliance resource.
    args (parser_extensions.Namespace): The args from the command.

  Returns:
    List[str] A list of strings representing the update mask.
  """
    update_mask = []
    if args.model is not None:
        appliance_resource.model = getattr(flags.APPLIANCE_MODEL_ENUM, args.model)
        update_mask.append('model')
    if args.IsSpecified('display_name'):
        appliance_resource.displayName = args.display_name
        update_mask.append('displayName')
    if args.IsSpecified('cmek'):
        appliance_resource.customerManagedKey = args.cmek
        update_mask.append('customerManagedKey')
    if hasattr(args, 'internet_enabled'):
        appliance_resource.internetEnabled = args.internet_enabled
    if args.offline_import is not None:
        destination = _get_gcs_destination_from_url_string(args.offline_import)
        appliance_resource.offlineImportFeature = destination
        update_mask.append('offlineImportFeature')
    if args.online_import is not None:
        destination = _get_gcs_destination_from_url_string(args.online_import)
        appliance_resource.onlineImportFeature = destination
        update_mask.append('onlineImportFeature')
    if args.offline_export is not None:
        offline_export = {'source': []}
        source = args.offline_export.get('source', None)
        manifest = args.offline_export.get('manifest', None)
        if source is not None:
            bucket, path = _get_bucket_folder_from_url_string(source)
            offline_export['source'].append({'bucket': '{}/{}'.format(bucket, path)})
        if manifest is not None:
            offline_export['transferManifest'] = {'location': manifest}
        appliance_resource.offlineExportFeature = offline_export
        update_mask.append('offlineExportFeature')
    return ','.join(update_mask)