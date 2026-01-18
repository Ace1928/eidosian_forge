import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.codedeploy import exceptions
def create_deployment_group(self, application_name, deployment_group_name, deployment_config_name=None, ec_2_tag_filters=None, auto_scaling_groups=None, service_role_arn=None):
    """
        Creates a new deployment group for application revisions to be
        deployed to.

        :type application_name: string
        :param application_name: The name of an existing AWS CodeDeploy
            application within the AWS user account.

        :type deployment_group_name: string
        :param deployment_group_name: The name of an existing deployment group
            for the specified application.

        :type deployment_config_name: string
        :param deployment_config_name: If specified, the deployment
            configuration name must be one of the predefined values, or it can
            be a custom deployment configuration:

        + CodeDeployDefault.AllAtOnce deploys an application revision to up to
              all of the Amazon EC2 instances at once. The overall deployment
              succeeds if the application revision deploys to at least one of the
              instances. The overall deployment fails after the application
              revision fails to deploy to all of the instances. For example, for
              9 instances, deploy to up to all 9 instances at once. The overall
              deployment succeeds if any of the 9 instances is successfully
              deployed to, and it fails if all 9 instances fail to be deployed
              to.
        + CodeDeployDefault.HalfAtATime deploys to up to half of the instances
              at a time (with fractions rounded down). The overall deployment
              succeeds if the application revision deploys to at least half of
              the instances (with fractions rounded up); otherwise, the
              deployment fails. For example, for 9 instances, deploy to up to 4
              instances at a time. The overall deployment succeeds if 5 or more
              instances are successfully deployed to; otherwise, the deployment
              fails. Note that the deployment may successfully deploy to some
              instances, even if the overall deployment fails.
        + CodeDeployDefault.OneAtATime deploys the application revision to only
              one of the instances at a time. The overall deployment succeeds if
              the application revision deploys to all of the instances. The
              overall deployment fails after the application revision first fails
              to deploy to any one instance. For example, for 9 instances, deploy
              to one instance at a time. The overall deployment succeeds if all 9
              instances are successfully deployed to, and it fails if any of one
              of the 9 instances fail to be deployed to. Note that the deployment
              may successfully deploy to some instances, even if the overall
              deployment fails. This is the default deployment configuration if a
              configuration isn't specified for either the deployment or the
              deployment group.


        To create a custom deployment configuration, call the create deployment
            configuration operation.

        :type ec_2_tag_filters: list
        :param ec_2_tag_filters: The Amazon EC2 tags to filter on.

        :type auto_scaling_groups: list
        :param auto_scaling_groups: A list of associated Auto Scaling groups.

        :type service_role_arn: string
        :param service_role_arn: A service role ARN that allows AWS CodeDeploy
            to act on the user's behalf when interacting with AWS services.

        """
    params = {'applicationName': application_name, 'deploymentGroupName': deployment_group_name}
    if deployment_config_name is not None:
        params['deploymentConfigName'] = deployment_config_name
    if ec_2_tag_filters is not None:
        params['ec2TagFilters'] = ec_2_tag_filters
    if auto_scaling_groups is not None:
        params['autoScalingGroups'] = auto_scaling_groups
    if service_role_arn is not None:
        params['serviceRoleArn'] = service_role_arn
    return self.make_request(action='CreateDeploymentGroup', body=json.dumps(params))