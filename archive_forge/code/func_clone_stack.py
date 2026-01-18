import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.opsworks import exceptions
def clone_stack(self, source_stack_id, service_role_arn, name=None, region=None, vpc_id=None, attributes=None, default_instance_profile_arn=None, default_os=None, hostname_theme=None, default_availability_zone=None, default_subnet_id=None, custom_json=None, configuration_manager=None, chef_configuration=None, use_custom_cookbooks=None, use_opsworks_security_groups=None, custom_cookbooks_source=None, default_ssh_key_name=None, clone_permissions=None, clone_app_ids=None, default_root_device_type=None):
    """
        Creates a clone of a specified stack. For more information,
        see `Clone a Stack`_.

        **Required Permissions**: To use this action, an IAM user must
        have an attached policy that explicitly grants permissions.
        For more information on user permissions, see `Managing User
        Permissions`_.

        :type source_stack_id: string
        :param source_stack_id: The source stack ID.

        :type name: string
        :param name: The cloned stack name.

        :type region: string
        :param region: The cloned stack AWS region, such as "us-east-1". For
            more information about AWS regions, see `Regions and Endpoints`_.

        :type vpc_id: string
        :param vpc_id: The ID of the VPC that the cloned stack is to be
            launched into. It must be in the specified region. All instances
            are launched into this VPC, and you cannot change the ID later.

        + If your account supports EC2 Classic, the default value is no VPC.
        + If your account does not support EC2 Classic, the default value is
              the default VPC for the specified region.


        If the VPC ID corresponds to a default VPC and you have specified
            either the `DefaultAvailabilityZone` or the `DefaultSubnetId`
            parameter only, AWS OpsWorks infers the value of the other
            parameter. If you specify neither parameter, AWS OpsWorks sets
            these parameters to the first valid Availability Zone for the
            specified region and the corresponding default VPC subnet ID,
            respectively.

        If you specify a nondefault VPC ID, note the following:


        + It must belong to a VPC in your account that is in the specified
              region.
        + You must specify a value for `DefaultSubnetId`.


        For more information on how to use AWS OpsWorks with a VPC, see
            `Running a Stack in a VPC`_. For more information on default VPC
            and EC2 Classic, see `Supported Platforms`_.

        :type attributes: map
        :param attributes: A list of stack attributes and values as key/value
            pairs to be added to the cloned stack.

        :type service_role_arn: string
        :param service_role_arn:
        The stack AWS Identity and Access Management (IAM) role, which allows
            AWS OpsWorks to work with AWS resources on your behalf. You must
            set this parameter to the Amazon Resource Name (ARN) for an
            existing IAM role. If you create a stack by using the AWS OpsWorks
            console, it creates the role for you. You can obtain an existing
            stack's IAM ARN programmatically by calling DescribePermissions.
            For more information about IAM ARNs, see `Using Identifiers`_.


        You must set this parameter to a valid service role ARN or the action
            will fail; there is no default value. You can specify the source
            stack's service role ARN, if you prefer, but you must do so
            explicitly.

        :type default_instance_profile_arn: string
        :param default_instance_profile_arn: The ARN of an IAM profile that is
            the default profile for all of the stack's EC2 instances. For more
            information about IAM ARNs, see `Using Identifiers`_.

        :type default_os: string
        :param default_os: The stacks's operating system, which must be set to
            one of the following.

        + Standard operating systems: an Amazon Linux version such as `Amazon
              Linux 2014.09`, `Ubuntu 12.04 LTS`, or `Ubuntu 14.04 LTS`.
        + Custom AMIs: `Custom`. You specify the custom AMI you want to use
              when you create instances.


        The default option is the current Amazon Linux version.

        :type hostname_theme: string
        :param hostname_theme: The stack's host name theme, with spaces are
            replaced by underscores. The theme is used to generate host names
            for the stack's instances. By default, `HostnameTheme` is set to
            `Layer_Dependent`, which creates host names by appending integers
            to the layer's short name. The other themes are:

        + `Baked_Goods`
        + `Clouds`
        + `European_Cities`
        + `Fruits`
        + `Greek_Deities`
        + `Legendary_Creatures_from_Japan`
        + `Planets_and_Moons`
        + `Roman_Deities`
        + `Scottish_Islands`
        + `US_Cities`
        + `Wild_Cats`


        To obtain a generated host name, call `GetHostNameSuggestion`, which
            returns a host name based on the current theme.

        :type default_availability_zone: string
        :param default_availability_zone: The cloned stack's default
            Availability Zone, which must be in the specified region. For more
            information, see `Regions and Endpoints`_. If you also specify a
            value for `DefaultSubnetId`, the subnet must be in the same zone.
            For more information, see the `VpcId` parameter description.

        :type default_subnet_id: string
        :param default_subnet_id: The stack's default VPC subnet ID. This
            parameter is required if you specify a value for the `VpcId`
            parameter. All instances are launched into this subnet unless you
            specify otherwise when you create the instance. If you also specify
            a value for `DefaultAvailabilityZone`, the subnet must be in that
            zone. For information on default values and when this parameter is
            required, see the `VpcId` parameter description.

        :type custom_json: string
        :param custom_json: A string that contains user-defined, custom JSON.
            It is used to override the corresponding default stack
            configuration JSON values. The string should be in the following
            format and must escape characters such as '"'.:
        `"{"key1": "value1", "key2": "value2",...}"`

        For more information on custom JSON, see `Use Custom JSON to Modify the
            Stack Configuration JSON`_

        :type configuration_manager: dict
        :param configuration_manager: The configuration manager. When you clone
            a stack we recommend that you use the configuration manager to
            specify the Chef version, 0.9, 11.4, or 11.10. The default value is
            currently 11.4.

        :type chef_configuration: dict
        :param chef_configuration: A `ChefConfiguration` object that specifies
            whether to enable Berkshelf and the Berkshelf version on Chef 11.10
            stacks. For more information, see `Create a New Stack`_.

        :type use_custom_cookbooks: boolean
        :param use_custom_cookbooks: Whether to use custom cookbooks.

        :type use_opsworks_security_groups: boolean
        :param use_opsworks_security_groups: Whether to associate the AWS
            OpsWorks built-in security groups with the stack's layers.
        AWS OpsWorks provides a standard set of built-in security groups, one
            for each layer, which are associated with layers by default. With
            `UseOpsworksSecurityGroups` you can instead provide your own custom
            security groups. `UseOpsworksSecurityGroups` has the following
            settings:


        + True - AWS OpsWorks automatically associates the appropriate built-in
              security group with each layer (default setting). You can associate
              additional security groups with a layer after you create it but you
              cannot delete the built-in security group.
        + False - AWS OpsWorks does not associate built-in security groups with
              layers. You must create appropriate EC2 security groups and
              associate a security group with each layer that you create.
              However, you can still manually associate a built-in security group
              with a layer on creation; custom security groups are required only
              for those layers that need custom settings.


        For more information, see `Create a New Stack`_.

        :type custom_cookbooks_source: dict
        :param custom_cookbooks_source: Contains the information required to
            retrieve an app or cookbook from a repository. For more
            information, see `Creating Apps`_ or `Custom Recipes and
            Cookbooks`_.

        :type default_ssh_key_name: string
        :param default_ssh_key_name: A default SSH key for the stack instances.
            You can override this value when you create or update an instance.

        :type clone_permissions: boolean
        :param clone_permissions: Whether to clone the source stack's
            permissions.

        :type clone_app_ids: list
        :param clone_app_ids: A list of source stack app IDs to be included in
            the cloned stack.

        :type default_root_device_type: string
        :param default_root_device_type: The default root device type. This
            value is used by default for all instances in the cloned stack, but
            you can override it when you create an instance. For more
            information, see `Storage for the Root Device`_.

        """
    params = {'SourceStackId': source_stack_id, 'ServiceRoleArn': service_role_arn}
    if name is not None:
        params['Name'] = name
    if region is not None:
        params['Region'] = region
    if vpc_id is not None:
        params['VpcId'] = vpc_id
    if attributes is not None:
        params['Attributes'] = attributes
    if default_instance_profile_arn is not None:
        params['DefaultInstanceProfileArn'] = default_instance_profile_arn
    if default_os is not None:
        params['DefaultOs'] = default_os
    if hostname_theme is not None:
        params['HostnameTheme'] = hostname_theme
    if default_availability_zone is not None:
        params['DefaultAvailabilityZone'] = default_availability_zone
    if default_subnet_id is not None:
        params['DefaultSubnetId'] = default_subnet_id
    if custom_json is not None:
        params['CustomJson'] = custom_json
    if configuration_manager is not None:
        params['ConfigurationManager'] = configuration_manager
    if chef_configuration is not None:
        params['ChefConfiguration'] = chef_configuration
    if use_custom_cookbooks is not None:
        params['UseCustomCookbooks'] = use_custom_cookbooks
    if use_opsworks_security_groups is not None:
        params['UseOpsworksSecurityGroups'] = use_opsworks_security_groups
    if custom_cookbooks_source is not None:
        params['CustomCookbooksSource'] = custom_cookbooks_source
    if default_ssh_key_name is not None:
        params['DefaultSshKeyName'] = default_ssh_key_name
    if clone_permissions is not None:
        params['ClonePermissions'] = clone_permissions
    if clone_app_ids is not None:
        params['CloneAppIds'] = clone_app_ids
    if default_root_device_type is not None:
        params['DefaultRootDeviceType'] = default_root_device_type
    return self.make_request(action='CloneStack', body=json.dumps(params))