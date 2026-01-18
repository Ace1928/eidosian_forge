import boto
from boto.compat import json, urlsplit
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.machinelearning import exceptions
def create_data_source_from_rds(self, data_source_id, rds_data, role_arn, data_source_name=None, compute_statistics=None):
    """
        Creates a `DataSource` object from an ` Amazon Relational
        Database Service`_ (Amazon RDS). A `DataSource` references
        data that can be used to perform CreateMLModel,
        CreateEvaluation, or CreateBatchPrediction operations.

        `CreateDataSourceFromRDS` is an asynchronous operation. In
        response to `CreateDataSourceFromRDS`, Amazon Machine Learning
        (Amazon ML) immediately returns and sets the `DataSource`
        status to `PENDING`. After the `DataSource` is created and
        ready for use, Amazon ML sets the `Status` parameter to
        `COMPLETED`. `DataSource` in `COMPLETED` or `PENDING` status
        can only be used to perform CreateMLModel, CreateEvaluation,
        or CreateBatchPrediction operations.

        If Amazon ML cannot accept the input source, it sets the
        `Status` parameter to `FAILED` and includes an error message
        in the `Message` attribute of the GetDataSource operation
        response.

        :type data_source_id: string
        :param data_source_id: A user-supplied ID that uniquely identifies the
            `DataSource`. Typically, an Amazon Resource Number (ARN) becomes
            the ID for a `DataSource`.

        :type data_source_name: string
        :param data_source_name: A user-supplied name or description of the
            `DataSource`.

        :type rds_data: dict
        :param rds_data:
        The data specification of an Amazon RDS `DataSource`:


        + DatabaseInformation -

            + `DatabaseName ` - Name of the Amazon RDS database.
            + ` InstanceIdentifier ` - Unique identifier for the Amazon RDS
                  database instance.

        + DatabaseCredentials - AWS Identity and Access Management (IAM)
              credentials that are used to connect to the Amazon RDS database.
        + ResourceRole - Role (DataPipelineDefaultResourceRole) assumed by an
              Amazon Elastic Compute Cloud (EC2) instance to carry out the copy
              task from Amazon RDS to Amazon S3. For more information, see `Role
              templates`_ for data pipelines.
        + ServiceRole - Role (DataPipelineDefaultRole) assumed by the AWS Data
              Pipeline service to monitor the progress of the copy task from
              Amazon RDS to Amazon Simple Storage Service (S3). For more
              information, see `Role templates`_ for data pipelines.
        + SecurityInfo - Security information to use to access an Amazon RDS
              instance. You need to set up appropriate ingress rules for the
              security entity IDs provided to allow access to the Amazon RDS
              instance. Specify a [ `SubnetId`, `SecurityGroupIds`] pair for a
              VPC-based Amazon RDS instance.
        + SelectSqlQuery - Query that is used to retrieve the observation data
              for the `Datasource`.
        + S3StagingLocation - Amazon S3 location for staging RDS data. The data
              retrieved from Amazon RDS using `SelectSqlQuery` is stored in this
              location.
        + DataSchemaUri - Amazon S3 location of the `DataSchema`.
        + DataSchema - A JSON string representing the schema. This is not
              required if `DataSchemaUri` is specified.
        + DataRearrangement - A JSON string representing the splitting
              requirement of a `Datasource`. Sample - ` "{"randomSeed":"some-
              random-seed",
              "splitting":{"percentBegin":10,"percentEnd":60}}"`

        :type role_arn: string
        :param role_arn: The role that Amazon ML assumes on behalf of the user
            to create and activate a data pipeline in the users account and
            copy data (using the `SelectSqlQuery`) query from Amazon RDS to
            Amazon S3.

        :type compute_statistics: boolean
        :param compute_statistics: The compute statistics for a `DataSource`.
            The statistics are generated from the observation data referenced
            by a `DataSource`. Amazon ML uses the statistics internally during
            an `MLModel` training. This parameter must be set to `True` if the
            ``DataSource `` needs to be used for `MLModel` training.

        """
    params = {'DataSourceId': data_source_id, 'RDSData': rds_data, 'RoleARN': role_arn}
    if data_source_name is not None:
        params['DataSourceName'] = data_source_name
    if compute_statistics is not None:
        params['ComputeStatistics'] = compute_statistics
    return self.make_request(action='CreateDataSourceFromRDS', body=json.dumps(params))