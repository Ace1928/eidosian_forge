import types
import boto
import boto.utils
from boto.ec2.regioninfo import RegionInfo
from boto.emr.emrobject import AddInstanceGroupsResponse, BootstrapActionList, \
from boto.emr.step import JarStep
from boto.connection import AWSQueryConnection
from boto.exception import EmrResponseError
from boto.compat import six
def run_jobflow(self, name, log_uri=None, ec2_keyname=None, availability_zone=None, master_instance_type='m1.small', slave_instance_type='m1.small', num_instances=1, action_on_failure='TERMINATE_JOB_FLOW', keep_alive=False, enable_debugging=False, hadoop_version=None, steps=None, bootstrap_actions=[], instance_groups=None, additional_info=None, ami_version=None, api_params=None, visible_to_all_users=None, job_flow_role=None, service_role=None):
    """
        Runs a job flow
        :type name: str
        :param name: Name of the job flow

        :type log_uri: str
        :param log_uri: URI of the S3 bucket to place logs

        :type ec2_keyname: str
        :param ec2_keyname: EC2 key used for the instances

        :type availability_zone: str
        :param availability_zone: EC2 availability zone of the cluster

        :type master_instance_type: str
        :param master_instance_type: EC2 instance type of the master

        :type slave_instance_type: str
        :param slave_instance_type: EC2 instance type of the slave nodes

        :type num_instances: int
        :param num_instances: Number of instances in the Hadoop cluster

        :type action_on_failure: str
        :param action_on_failure: Action to take if a step terminates

        :type keep_alive: bool
        :param keep_alive: Denotes whether the cluster should stay
            alive upon completion

        :type enable_debugging: bool
        :param enable_debugging: Denotes whether AWS console debugging
            should be enabled.

        :type hadoop_version: str
        :param hadoop_version: Version of Hadoop to use. This no longer
            defaults to '0.20' and now uses the AMI default.

        :type steps: list(boto.emr.Step)
        :param steps: List of steps to add with the job

        :type bootstrap_actions: list(boto.emr.BootstrapAction)
        :param bootstrap_actions: List of bootstrap actions that run
            before Hadoop starts.

        :type instance_groups: list(boto.emr.InstanceGroup)
        :param instance_groups: Optional list of instance groups to
            use when creating this job.
            NB: When provided, this argument supersedes num_instances
            and master/slave_instance_type.

        :type ami_version: str
        :param ami_version: Amazon Machine Image (AMI) version to use
            for instances. Values accepted by EMR are '1.0', '2.0', and
            'latest'; EMR currently defaults to '1.0' if you don't set
            'ami_version'.

        :type additional_info: JSON str
        :param additional_info: A JSON string for selecting additional features

        :type api_params: dict
        :param api_params: a dictionary of additional parameters to pass
            directly to the EMR API (so you don't have to upgrade boto to
            use new EMR features). You can also delete an API parameter
            by setting it to None.

        :type visible_to_all_users: bool
        :param visible_to_all_users: Whether the job flow is visible to all IAM
            users of the AWS account associated with the job flow. If this
            value is set to ``True``, all IAM users of that AWS
            account can view and (if they have the proper policy permissions
            set) manage the job flow. If it is set to ``False``, only
            the IAM user that created the job flow can view and manage
            it.

        :type job_flow_role: str
        :param job_flow_role: An IAM role for the job flow. The EC2
            instances of the job flow assume this role. The default role is
            ``EMRJobflowDefault``. In order to use the default role,
            you must have already created it using the CLI.

        :type service_role: str
        :param service_role: The IAM role that will be assumed by the Amazon
            EMR service to access AWS resources on your behalf.

        :rtype: str
        :return: The jobflow id
        """
    steps = steps or []
    params = {}
    if action_on_failure:
        params['ActionOnFailure'] = action_on_failure
    if log_uri:
        params['LogUri'] = log_uri
    params['Name'] = name
    common_params = self._build_instance_common_args(ec2_keyname, availability_zone, keep_alive, hadoop_version)
    params.update(common_params)
    if not instance_groups:
        instance_params = self._build_instance_count_and_type_args(master_instance_type, slave_instance_type, num_instances)
        params.update(instance_params)
    else:
        list_args = self._build_instance_group_list_args(instance_groups)
        instance_params = dict((('Instances.%s' % k, v) for k, v in six.iteritems(list_args)))
        params.update(instance_params)
    if enable_debugging:
        debugging_step = JarStep(name='Setup Hadoop Debugging', action_on_failure='TERMINATE_JOB_FLOW', main_class=None, jar=self.DebuggingJar.format(region_name=self.region.name), step_args=self.DebuggingArgs.format(region_name=self.region.name))
        steps.insert(0, debugging_step)
    if steps:
        step_args = [self._build_step_args(step) for step in steps]
        params.update(self._build_step_list(step_args))
    if bootstrap_actions:
        bootstrap_action_args = [self._build_bootstrap_action_args(bootstrap_action) for bootstrap_action in bootstrap_actions]
        params.update(self._build_bootstrap_action_list(bootstrap_action_args))
    if ami_version:
        params['AmiVersion'] = ami_version
    if additional_info is not None:
        params['AdditionalInfo'] = additional_info
    if api_params:
        for key, value in six.iteritems(api_params):
            if value is None:
                params.pop(key, None)
            else:
                params[key] = value
    if visible_to_all_users is not None:
        if visible_to_all_users:
            params['VisibleToAllUsers'] = 'true'
        else:
            params['VisibleToAllUsers'] = 'false'
    if job_flow_role is not None:
        params['JobFlowRole'] = job_flow_role
    if service_role is not None:
        params['ServiceRole'] = service_role
    response = self.get_object('RunJobFlow', params, RunJobFlowResponse, verb='POST')
    return response.jobflowid