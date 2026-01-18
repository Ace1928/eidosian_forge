import boto
import boto.jsonresponse
from boto.compat import json
from boto.regioninfo import RegionInfo
from boto.connection import AWSQueryConnection
def create_application_version(self, application_name, version_label, description=None, s3_bucket=None, s3_key=None, auto_create_application=None):
    """Creates an application version for the specified application.

        :type application_name: string
        :param application_name: The name of the application. If no
            application is found with this name, and AutoCreateApplication is
            false, returns an InvalidParameterValue error.

        :type version_label: string
        :param version_label: A label identifying this version. Constraint:
            Must be unique per application. If an application version already
            exists with this label for the specified application, AWS Elastic
            Beanstalk returns an InvalidParameterValue error.

        :type description: string
        :param description: Describes this version.

        :type s3_bucket: string
        :param s3_bucket: The Amazon S3 bucket where the data is located.

        :type s3_key: string
        :param s3_key: The Amazon S3 key where the data is located.  Both
            s3_bucket and s3_key must be specified in order to use a specific
            source bundle.  If both of these values are not specified the
            sample application will be used.

        :type auto_create_application: boolean
        :param auto_create_application: Determines how the system behaves if
            the specified application for this version does not already exist:
            true: Automatically creates the specified application for this
            version if it does not already exist.  false: Returns an
            InvalidParameterValue if the specified application for this version
            does not already exist.  Default: false  Valid Values: true | false

        :raises: TooManyApplicationsException,
                 TooManyApplicationVersionsException,
                 InsufficientPrivilegesException,
                 S3LocationNotInServiceRegionException

        """
    params = {'ApplicationName': application_name, 'VersionLabel': version_label}
    if description:
        params['Description'] = description
    if s3_bucket and s3_key:
        params['SourceBundle.S3Bucket'] = s3_bucket
        params['SourceBundle.S3Key'] = s3_key
    if auto_create_application:
        params['AutoCreateApplication'] = self._encode_bool(auto_create_application)
    return self._get_response('CreateApplicationVersion', params)