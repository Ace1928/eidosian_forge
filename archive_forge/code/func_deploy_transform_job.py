import json
import os
import tempfile
import click
import mlflow
import mlflow.models.docker_utils
import mlflow.sagemaker
from mlflow.sagemaker import DEFAULT_IMAGE_NAME as IMAGE
from mlflow.utils import cli_args
from mlflow.utils import env_manager as em
@commands.command('deploy-transform-job')
@click.option('--job-name', '-n', help='Transform job name', required=True)
@cli_args.MODEL_URI
@click.option('--input-data-type', help='Input data type for the transform job', required=True)
@click.option('--input-uri', '-u', help='S3 key name prefix or manifest of the input data', required=True)
@click.option('--content-type', help='The multipurpose internet mail extension (MIME) type of the data', required=True)
@click.option('--output-path', '-o', help='The S3 path to store the output results of the Sagemaker transform job', required=True)
@click.option('--compression-type', default='None', help='The compression type of the transform data')
@click.option('--split-type', '-s', default='Line', help="The method to split the transform job's data files into smaller batches")
@click.option('--accept', '-a', default='text/csv', help='The multipurpose internet mail extension (MIME) type of the output data')
@click.option('--assemble-with', default='Line', help='The method to assemble the results of the transform job as a single S3 object')
@click.option('--input-filter', default='$', help='A JSONPath expression used to select a portion of the input data for the transform job')
@click.option('--output-filter', default='$', help='A JSONPath expression used to select a portion of the output data from the transform job')
@click.option('--join-resource', '-j', default='None', help='The source of the data to join with the transformed data')
@click.option('--execution-role-arn', '-e', default=None, help='SageMaker execution role')
@click.option('--bucket', '-b', default=None, help='S3 bucket to store model artifacts')
@click.option('--image-url', '-i', default=None, help='ECR URL for the Docker image')
@click.option('--region-name', default='us-west-2', help='Name of the AWS region in which to deploy the transform job')
@click.option('--instance-type', '-t', default=mlflow.sagemaker.DEFAULT_SAGEMAKER_INSTANCE_TYPE, help='The type of SageMaker ML instance on which to perform the batch transform job. For a list of supported instance types, see https://aws.amazon.com/sagemaker/pricing/instance-types/.')
@click.option('--instance-count', '-c', default=mlflow.sagemaker.DEFAULT_SAGEMAKER_INSTANCE_COUNT, help='The number of SageMaker ML instances on which to perform the batch transform job')
@click.option('--vpc-config', '-v', help='Path to a file containing a JSON-formatted VPC configuration. This configuration will be used when creating the new SageMaker model associated with this application. For more information, see https://docs.aws.amazon.com/sagemaker/latest/dg/API_VpcConfig.html')
@click.option('--flavor', '-f', default=None, help=f"The name of the flavor to use for deployment. Must be one of the following: {mlflow.sagemaker.SUPPORTED_DEPLOYMENT_FLAVORS}. If unspecified, a flavor will be automatically selected from the model's available flavors.")
@click.option('--archive', is_flag=True, help='If specified, any SageMaker resources that become inactive after the finished batch transform job are preserved. These resources may include the associated SageMaker models and model artifacts. Otherwise, if `--archive` is unspecified, these resources are deleted. `--archive` must be specified when deploying asynchronously with `--async`.')
@click.option('--async', 'asynchronous', is_flag=True, help='If specified, this command will return immediately after starting the deployment process. It will not wait for the deployment process to complete. The caller is responsible for monitoring the deployment process via native SageMaker APIs or the AWS console.')
@click.option('--timeout', default=1200, help='If the command is executed synchronously, the deployment process will return after the specified number of seconds if no definitive result (success or failure) is achieved. Once the function returns, the caller is responsible for monitoring the health and status of the pending deployment via native SageMaker APIs or the AWS console. If the command is executed asynchronously using the `--async` flag, this value is ignored.')
def deploy_transform_job(job_name, model_uri, input_data_type, input_uri, content_type, output_path, compression_type, split_type, accept, assemble_with, input_filter, output_filter, join_resource, execution_role_arn, bucket, image_url, region_name, instance_type, instance_count, vpc_config, flavor, archive, asynchronous, timeout):
    """
    Deploy model on Sagemaker as a batch transform job. Current active AWS account needs to have
    correct permissions setup.

    By default, unless the ``--async`` flag is specified, this command will block until
    either the batch transform job completes (definitively succeeds or fails) or the specified
    timeout elapses.
    """
    if vpc_config is not None:
        with open(vpc_config) as f:
            vpc_config = json.load(f)
    mlflow.sagemaker.deploy_transform_job(job_name=job_name, model_uri=model_uri, s3_input_data_type=input_data_type, s3_input_uri=input_uri, content_type=content_type, s3_output_path=output_path, compression_type=compression_type, split_type=split_type, accept=accept, assemble_with=assemble_with, input_filter=input_filter, output_filter=output_filter, join_resource=join_resource, execution_role_arn=execution_role_arn, bucket=bucket, image_url=image_url, region_name=region_name, instance_type=instance_type, instance_count=instance_count, vpc_config=vpc_config, flavor=flavor, archive=archive, synchronous=not asynchronous, timeout_seconds=timeout)