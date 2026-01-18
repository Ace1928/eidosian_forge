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
@commands.command('terminate-transform-job')
@click.option('--job-name', '-n', help='Transform job name', required=True)
@click.option('--region-name', '-r', default='us-west-2', help='Name of the AWS region in which the transform job is deployed')
@click.option('--archive', is_flag=True, help='If specified, resources associated with the application are preserved. These resources may include unused SageMaker models and model artifacts. Otherwise, if `--archive` is unspecified, these resources are deleted. `--archive` must be specified when deleting asynchronously with `--async`.')
@click.option('--async', 'asynchronous', is_flag=True, help='If specified, this command will return immediately after starting the termination process. It will not wait for the termination process to complete. The caller is responsible for monitoring the termination process via native SageMaker APIs or the AWS console.')
@click.option('--timeout', default=1200, help='If the command is executed synchronously, the termination process will return after the specified number of seconds if no definitive result (success or failure) is achieved. Once the function returns, the caller is responsible for monitoring the health and status of the pending termination via native SageMaker APIs or the AWS console. If the command is executed asynchronously using the `--async` flag, this value is ignored.')
def terminate_transform_job(job_name, region_name, archive, asynchronous, timeout):
    """
    Terminate the specified Sagemaker batch transform job. Unless ``--archive`` is specified,
    all SageMaker resources associated with the batch transform job are deleted as well.

    By default, unless the ``--async`` flag is specified, this command will block until
    either the termination process completes (definitively succeeds or fails) or the specified
    timeout elapses.
    """
    mlflow.sagemaker.terminate_transform_job(job_name=job_name, region_name=region_name, archive=archive, synchronous=not asynchronous, timeout_seconds=timeout)