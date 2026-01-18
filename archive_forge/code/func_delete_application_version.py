import boto
import boto.jsonresponse
from boto.compat import json
from boto.regioninfo import RegionInfo
from boto.connection import AWSQueryConnection
def delete_application_version(self, application_name, version_label, delete_source_bundle=None):
    """Deletes the specified version from the specified application.

        :type application_name: string
        :param application_name: The name of the application to delete
            releases from.

        :type version_label: string
        :param version_label: The label of the version to delete.

        :type delete_source_bundle: boolean
        :param delete_source_bundle: Indicates whether to delete the
            associated source bundle from Amazon S3.  Valid Values: true |
            false

        :raises: SourceBundleDeletionException,
                 InsufficientPrivilegesException,
                 OperationInProgressException,
                 S3LocationNotInServiceRegionException
        """
    params = {'ApplicationName': application_name, 'VersionLabel': version_label}
    if delete_source_bundle:
        params['DeleteSourceBundle'] = self._encode_bool(delete_source_bundle)
    return self._get_response('DeleteApplicationVersion', params)