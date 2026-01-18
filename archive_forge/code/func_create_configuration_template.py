import boto
import boto.jsonresponse
from boto.compat import json
from boto.regioninfo import RegionInfo
from boto.connection import AWSQueryConnection
def create_configuration_template(self, application_name, template_name, solution_stack_name=None, source_configuration_application_name=None, source_configuration_template_name=None, environment_id=None, description=None, option_settings=None):
    """Creates a configuration template.

        Templates are associated with a specific application and are used to
        deploy different versions of the application with the same
        configuration settings.

        :type application_name: string
        :param application_name: The name of the application to associate with
            this configuration template. If no application is found with this
            name, AWS Elastic Beanstalk returns an InvalidParameterValue error.

        :type template_name: string
        :param template_name: The name of the configuration template.
            Constraint: This name must be unique per application.  Default: If
            a configuration template already exists with this name, AWS Elastic
            Beanstalk returns an InvalidParameterValue error.

        :type solution_stack_name: string
        :param solution_stack_name: The name of the solution stack used by this
            configuration. The solution stack specifies the operating system,
            architecture, and application server for a configuration template.
            It determines the set of configuration options as well as the
            possible and default values.  Use ListAvailableSolutionStacks to
            obtain a list of available solution stacks.  Default: If the
            SolutionStackName is not specified and the source configuration
            parameter is blank, AWS Elastic Beanstalk uses the default solution
            stack. If not specified and the source configuration parameter is
            specified, AWS Elastic Beanstalk uses the same solution stack as
            the source configuration template.

        :type source_configuration_application_name: string
        :param source_configuration_application_name: The name of the
            application associated with the configuration.

        :type source_configuration_template_name: string
        :param source_configuration_template_name: The name of the
            configuration template.

        :type environment_id: string
        :param environment_id: The ID of the environment used with this
            configuration template.

        :type description: string
        :param description: Describes this configuration.

        :type option_settings: list
        :param option_settings: If specified, AWS Elastic Beanstalk sets the
            specified configuration option to the requested value. The new
            value overrides the value obtained from the solution stack or the
            source configuration template.

        :raises: InsufficientPrivilegesException,
                 TooManyConfigurationTemplatesException
        """
    params = {'ApplicationName': application_name, 'TemplateName': template_name}
    if solution_stack_name:
        params['SolutionStackName'] = solution_stack_name
    if source_configuration_application_name:
        params['SourceConfiguration.ApplicationName'] = source_configuration_application_name
    if source_configuration_template_name:
        params['SourceConfiguration.TemplateName'] = source_configuration_template_name
    if environment_id:
        params['EnvironmentId'] = environment_id
    if description:
        params['Description'] = description
    if option_settings:
        self._build_list_params(params, option_settings, 'OptionSettings.member', ('Namespace', 'OptionName', 'Value'))
    return self._get_response('CreateConfigurationTemplate', params)