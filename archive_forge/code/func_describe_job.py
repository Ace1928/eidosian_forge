import os
import boto.glacier
from boto.compat import json
from boto.connection import AWSAuthConnection
from boto.glacier.exceptions import UnexpectedHTTPResponseError
from boto.glacier.response import GlacierResponse
from boto.glacier.utils import ResettingFileSender
def describe_job(self, vault_name, job_id):
    """
        This operation returns information about a job you previously
        initiated, including the job initiation date, the user who
        initiated the job, the job status code/message and the Amazon
        SNS topic to notify after Amazon Glacier completes the job.
        For more information about initiating a job, see InitiateJob.


        This operation enables you to check the status of your job.
        However, it is strongly recommended that you set up an Amazon
        SNS topic and specify it in your initiate job request so that
        Amazon Glacier can notify the topic after it completes the
        job.


        A job ID will not expire for at least 24 hours after Amazon
        Glacier completes the job.

        An AWS account has full permission to perform all operations
        (actions). However, AWS Identity and Access Management (IAM)
        users don't have any permissions by default. You must grant
        them explicit permission to perform specific actions. For more
        information, see `Access Control Using AWS Identity and Access
        Management (IAM)`_.

        For information about the underlying REST API, go to `Working
        with Archives in Amazon Glacier`_ in the Amazon Glacier
        Developer Guide .

        :type vault_name: string
        :param vault_name: The name of the vault.

        :type job_id: string
        :param job_id: The ID of the job to describe.
        """
    uri = 'vaults/%s/jobs/%s' % (vault_name, job_id)
    return self.make_request('GET', uri, ok_responses=(200,))