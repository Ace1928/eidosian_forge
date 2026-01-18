import boto
import boto.jsonresponse
from boto.compat import json
from boto.regioninfo import RegionInfo
from boto.connection import AWSQueryConnection
def describe_application_versions(self, application_name=None, version_labels=None):
    """Returns descriptions for existing application versions.

        :type application_name: string
        :param application_name: If specified, AWS Elastic Beanstalk restricts
            the returned descriptions to only include ones that are associated
            with the specified application.

        :type version_labels: list
        :param version_labels: If specified, restricts the returned
            descriptions to only include ones that have the specified version
            labels.

        """
    params = {}
    if application_name:
        params['ApplicationName'] = application_name
    if version_labels:
        self.build_list_params(params, version_labels, 'VersionLabels.member')
    return self._get_response('DescribeApplicationVersions', params)