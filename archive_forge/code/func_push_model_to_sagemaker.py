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
@commands.command('push-model')
@click.option('--model-name', '-n', help='Sagemaker model name', required=True)
@cli_args.MODEL_URI
@click.option('--execution-role-arn', '-e', default=None, help='SageMaker execution role')
@click.option('--bucket', '-b', default=None, help='S3 bucket to store model artifacts')
@click.option('--image-url', '-i', default=None, help='ECR URL for the Docker image')
@click.option('--region-name', default='us-west-2', help='Name of the AWS region in which to push the Sagemaker model')
@click.option('--vpc-config', '-v', help='Path to a file containing a JSON-formatted VPC configuration. This configuration will be used when creating the new SageMaker model. For more information, see https://docs.aws.amazon.com/sagemaker/latest/dg/API_VpcConfig.html')
@click.option('--flavor', '-f', default=None, help="The name of the flavor to use for deployment. Must be one of the following: {supported_flavors}. If unspecified, a flavor will be automatically selected from the model's available flavors.".format(supported_flavors=mlflow.sagemaker.SUPPORTED_DEPLOYMENT_FLAVORS))
def push_model_to_sagemaker(model_name, model_uri, execution_role_arn, bucket, image_url, region_name, vpc_config, flavor):
    """
    Push an MLflow model to Sagemaker model registry. Current active AWS account needs to have
    correct permissions setup.
    """
    if vpc_config is not None:
        with open(vpc_config) as f:
            vpc_config = json.load(f)
    mlflow.sagemaker.push_model_to_sagemaker(model_name=model_name, model_uri=model_uri, execution_role_arn=execution_role_arn, bucket=bucket, image_url=image_url, region_name=region_name, vpc_config=vpc_config, flavor=flavor)