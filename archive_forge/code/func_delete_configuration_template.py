import boto
import boto.jsonresponse
from boto.compat import json
from boto.regioninfo import RegionInfo
from boto.connection import AWSQueryConnection
def delete_configuration_template(self, application_name, template_name):
    """Deletes the specified configuration template.

        :type application_name: string
        :param application_name: The name of the application to delete
            the configuration template from.

        :type template_name: string
        :param template_name: The name of the configuration template to
            delete.

        :raises: OperationInProgressException

        """
    params = {'ApplicationName': application_name, 'TemplateName': template_name}
    return self._get_response('DeleteConfigurationTemplate', params)