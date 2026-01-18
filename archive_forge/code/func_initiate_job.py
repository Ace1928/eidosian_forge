import os
import boto.glacier
from boto.compat import json
from boto.connection import AWSAuthConnection
from boto.glacier.exceptions import UnexpectedHTTPResponseError
from boto.glacier.response import GlacierResponse
from boto.glacier.utils import ResettingFileSender
def initiate_job(self, vault_name, job_data):
    """
        This operation initiates a job of the specified type. In this
        release, you can initiate a job to retrieve either an archive
        or a vault inventory (a list of archives in a vault).

        Retrieving data from Amazon Glacier is a two-step process:


        #. Initiate a retrieval job.
        #. After the job completes, download the bytes.


        The retrieval request is executed asynchronously. When you
        initiate a retrieval job, Amazon Glacier creates a job and
        returns a job ID in the response. When Amazon Glacier
        completes the job, you can get the job output (archive or
        inventory data). For information about getting job output, see
        GetJobOutput operation.

        The job must complete before you can get its output. To
        determine when a job is complete, you have the following
        options:


        + **Use Amazon SNS Notification** You can specify an Amazon
          Simple Notification Service (Amazon SNS) topic to which Amazon
          Glacier can post a notification after the job is completed.
          You can specify an SNS topic per job request. The notification
          is sent only after Amazon Glacier completes the job. In
          addition to specifying an SNS topic per job request, you can
          configure vault notifications for a vault so that job
          notifications are always sent. For more information, see
          SetVaultNotifications.
        + **Get job details** You can make a DescribeJob request to
          obtain job status information while a job is in progress.
          However, it is more efficient to use an Amazon SNS
          notification to determine when a job is complete.



        The information you get via notification is same that you get
        by calling DescribeJob.


        If for a specific event, you add both the notification
        configuration on the vault and also specify an SNS topic in
        your initiate job request, Amazon Glacier sends both
        notifications. For more information, see
        SetVaultNotifications.

        An AWS account has full permission to perform all operations
        (actions). However, AWS Identity and Access Management (IAM)
        users don't have any permissions by default. You must grant
        them explicit permission to perform specific actions. For more
        information, see `Access Control Using AWS Identity and Access
        Management (IAM)`_.

        **About the Vault Inventory**

        Amazon Glacier prepares an inventory for each vault
        periodically, every 24 hours. When you initiate a job for a
        vault inventory, Amazon Glacier returns the last inventory for
        the vault. The inventory data you get might be up to a day or
        two days old. Also, the initiate inventory job might take some
        time to complete before you can download the vault inventory.
        So you do not want to retrieve a vault inventory for each
        vault operation. However, in some scenarios, you might find
        the vault inventory useful. For example, when you upload an
        archive, you can provide an archive description but not an
        archive name. Amazon Glacier provides you a unique archive ID,
        an opaque string of characters. So, you might maintain your
        own database that maps archive names to their corresponding
        Amazon Glacier assigned archive IDs. You might find the vault
        inventory useful in the event you need to reconcile
        information in your database with the actual vault inventory.

        **About Ranged Archive Retrieval**

        You can initiate an archive retrieval for the whole archive or
        a range of the archive. In the case of ranged archive
        retrieval, you specify a byte range to return or the whole
        archive. The range specified must be megabyte (MB) aligned,
        that is the range start value must be divisible by 1 MB and
        range end value plus 1 must be divisible by 1 MB or equal the
        end of the archive. If the ranged archive retrieval is not
        megabyte aligned, this operation returns a 400 response.
        Furthermore, to ensure you get checksum values for data you
        download using Get Job Output API, the range must be tree hash
        aligned.

        An AWS account has full permission to perform all operations
        (actions). However, AWS Identity and Access Management (IAM)
        users don't have any permissions by default. You must grant
        them explicit permission to perform specific actions. For more
        information, see `Access Control Using AWS Identity and Access
        Management (IAM)`_.

        For conceptual information and the underlying REST API, go to
        `Initiate a Job`_ and `Downloading a Vault Inventory`_

        :type account_id: string
        :param account_id: The `AccountId` is the AWS Account ID. You can
            specify either the AWS Account ID or optionally a '-', in which
            case Amazon Glacier uses the AWS Account ID associated with the
            credentials used to sign the request. If you specify your Account
            ID, do not include hyphens in it.

        :type vault_name: string
        :param vault_name: The name of the vault.

        :type job_parameters: dict
        :param job_parameters: Provides options for specifying job information.
            The dictionary can contain the following attributes:

            * ArchiveId - The ID of the archive you want to retrieve.
              This field is required only if the Type is set to
              archive-retrieval.
            * Description - The optional description for the job.
            * Format - When initiating a job to retrieve a vault
              inventory, you can optionally add this parameter to
              specify the output format.  Valid values are: CSV|JSON.
            * SNSTopic - The Amazon SNS topic ARN where Amazon Glacier
              sends a notification when the job is completed and the
              output is ready for you to download.
            * Type - The job type.  Valid values are:
              archive-retrieval|inventory-retrieval
            * RetrievalByteRange - Optionally specify the range of
              bytes to retrieve.
            * InventoryRetrievalParameters: Optional job parameters
                * Format - The output format, like "JSON"
                * StartDate - ISO8601 starting date string
                * EndDate - ISO8601 ending date string
                * Limit - Maximum number of entries
                * Marker - A unique string used for pagination

        """
    uri = 'vaults/%s/jobs' % vault_name
    response_headers = [('x-amz-job-id', u'JobId'), ('Location', u'Location')]
    json_job_data = json.dumps(job_data)
    return self.make_request('POST', uri, data=json_job_data, ok_responses=(202,), response_headers=response_headers)