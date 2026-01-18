import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.codedeploy import exceptions
def register_application_revision(self, application_name, revision, description=None):
    """
        Registers with AWS CodeDeploy a revision for the specified
        application.

        :type application_name: string
        :param application_name: The name of an existing AWS CodeDeploy
            application within the AWS user account.

        :type description: string
        :param description: A comment about the revision.

        :type revision: dict
        :param revision: Information about the application revision to
            register, including the revision's type and its location.

        """
    params = {'applicationName': application_name, 'revision': revision}
    if description is not None:
        params['description'] = description
    return self.make_request(action='RegisterApplicationRevision', body=json.dumps(params))