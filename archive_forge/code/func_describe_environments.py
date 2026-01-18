import boto
import boto.jsonresponse
from boto.compat import json
from boto.regioninfo import RegionInfo
from boto.connection import AWSQueryConnection
def describe_environments(self, application_name=None, version_label=None, environment_ids=None, environment_names=None, include_deleted=None, included_deleted_back_to=None):
    """Returns descriptions for existing environments.

        :type application_name: string
        :param application_name: If specified, AWS Elastic Beanstalk restricts
            the returned descriptions to include only those that are associated
            with this application.

        :type version_label: string
        :param version_label: If specified, AWS Elastic Beanstalk restricts the
            returned descriptions to include only those that are associated
            with this application version.

        :type environment_ids: list
        :param environment_ids: If specified, AWS Elastic Beanstalk restricts
            the returned descriptions to include only those that have the
            specified IDs.

        :type environment_names: list
        :param environment_names: If specified, AWS Elastic Beanstalk restricts
            the returned descriptions to include only those that have the
            specified names.

        :type include_deleted: boolean
        :param include_deleted: Indicates whether to include deleted
            environments:  true: Environments that have been deleted after
            IncludedDeletedBackTo are displayed.  false: Do not include deleted
            environments.

        :type included_deleted_back_to: timestamp
        :param included_deleted_back_to: If specified when IncludeDeleted is
            set to true, then environments deleted after this date are
            displayed.
        """
    params = {}
    if application_name:
        params['ApplicationName'] = application_name
    if version_label:
        params['VersionLabel'] = version_label
    if environment_ids:
        self.build_list_params(params, environment_ids, 'EnvironmentIds.member')
    if environment_names:
        self.build_list_params(params, environment_names, 'EnvironmentNames.member')
    if include_deleted:
        params['IncludeDeleted'] = self._encode_bool(include_deleted)
    if included_deleted_back_to:
        params['IncludedDeletedBackTo'] = included_deleted_back_to
    return self._get_response('DescribeEnvironments', params)