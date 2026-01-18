import boto
from boto.compat import json, urlsplit
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.machinelearning import exceptions
def create_data_source_from_redshift(self, data_source_id, data_spec, role_arn, data_source_name=None, compute_statistics=None):
    """
        Creates a `DataSource` from `Amazon Redshift`_. A `DataSource`
        references data that can be used to perform either
        CreateMLModel, CreateEvaluation or CreateBatchPrediction
        operations.

        `CreateDataSourceFromRedshift` is an asynchronous operation.
        In response to `CreateDataSourceFromRedshift`, Amazon Machine
        Learning (Amazon ML) immediately returns and sets the
        `DataSource` status to `PENDING`. After the `DataSource` is
        created and ready for use, Amazon ML sets the `Status`
        parameter to `COMPLETED`. `DataSource` in `COMPLETED` or
        `PENDING` status can only be used to perform CreateMLModel,
        CreateEvaluation, or CreateBatchPrediction operations.

        If Amazon ML cannot accept the input source, it sets the
        `Status` parameter to `FAILED` and includes an error message
        in the `Message` attribute of the GetDataSource operation
        response.

        The observations should exist in the database hosted on an
        Amazon Redshift cluster and should be specified by a
        `SelectSqlQuery`. Amazon ML executes ` Unload`_ command in
        Amazon Redshift to transfer the result set of `SelectSqlQuery`
        to `S3StagingLocation.`

        After the `DataSource` is created, it's ready for use in
        evaluations and batch predictions. If you plan to use the
        `DataSource` to train an `MLModel`, the `DataSource` requires
        another item -- a recipe. A recipe describes the observation
        variables that participate in training an `MLModel`. A recipe
        describes how each input variable will be used in training.
        Will the variable be included or excluded from training? Will
        the variable be manipulated, for example, combined with
        another variable or split apart into word combinations? The
        recipe provides answers to these questions. For more
        information, see the Amazon Machine Learning Developer Guide.

        :type data_source_id: string
        :param data_source_id: A user-supplied ID that uniquely identifies the
            `DataSource`.

        :type data_source_name: string
        :param data_source_name: A user-supplied name or description of the
            `DataSource`.

        :type data_spec: dict
        :param data_spec:
        The data specification of an Amazon Redshift `DataSource`:


        + DatabaseInformation -

            + `DatabaseName ` - Name of the Amazon Redshift database.
            + ` ClusterIdentifier ` - Unique ID for the Amazon Redshift cluster.

        + DatabaseCredentials - AWS Identity abd Access Management (IAM)
              credentials that are used to connect to the Amazon Redshift
              database.
        + SelectSqlQuery - Query that is used to retrieve the observation data
              for the `Datasource`.
        + S3StagingLocation - Amazon Simple Storage Service (Amazon S3)
              location for staging Amazon Redshift data. The data retrieved from
              Amazon Relational Database Service (Amazon RDS) using
              `SelectSqlQuery` is stored in this location.
        + DataSchemaUri - Amazon S3 location of the `DataSchema`.
        + DataSchema - A JSON string representing the schema. This is not
              required if `DataSchemaUri` is specified.
        + DataRearrangement - A JSON string representing the splitting
              requirement of a `Datasource`. Sample - ` "{"randomSeed":"some-
              random-seed",
              "splitting":{"percentBegin":10,"percentEnd":60}}"`

        :type role_arn: string
        :param role_arn: A fully specified role Amazon Resource Name (ARN).
            Amazon ML assumes the role on behalf of the user to create the
            following:


        + A security group to allow Amazon ML to execute the `SelectSqlQuery`
              query on an Amazon Redshift cluster
        + An Amazon S3 bucket policy to grant Amazon ML read/write permissions
              on the `S3StagingLocation`

        :type compute_statistics: boolean
        :param compute_statistics: The compute statistics for a `DataSource`.
            The statistics are generated from the observation data referenced
            by a `DataSource`. Amazon ML uses the statistics internally during
            `MLModel` training. This parameter must be set to `True` if the
            ``DataSource `` needs to be used for `MLModel` training

        """
    params = {'DataSourceId': data_source_id, 'DataSpec': data_spec, 'RoleARN': role_arn}
    if data_source_name is not None:
        params['DataSourceName'] = data_source_name
    if compute_statistics is not None:
        params['ComputeStatistics'] = compute_statistics
    return self.make_request(action='CreateDataSourceFromRedshift', body=json.dumps(params))