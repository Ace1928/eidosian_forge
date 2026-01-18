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
@commands.command('build-and-push-container')
@click.option('--build/--no-build', default=True, help='Build the container if set.')
@click.option('--push/--no-push', default=True, help='Push the container to AWS ECR if set.')
@click.option('--container', '-c', default=IMAGE, help='image name')
@cli_args.ENV_MANAGER
@cli_args.MLFLOW_HOME
def build_and_push_container(build, push, container, env_manager, mlflow_home):
    """
    Build new MLflow Sagemaker image, assign it a name, and push to ECR.

    This function builds an MLflow Docker image.
    The image is built locally and it requires Docker to run.
    The image is pushed to ECR under current active AWS account and to current active AWS region.
    """
    from mlflow.models import docker_utils
    env_manager = env_manager or em.VIRTUALENV
    if not (build or push):
        click.echo('skipping both build and push, have nothing to do!')
    if build:
        sagemaker_image_entrypoint = f"import sys; from mlflow.models import container as C; C._init(sys.argv[1], '{env_manager}')"
        setup_container = '# Install minimal serving dependencies\nRUN python -c "from mlflow.models.container import _install_pyfunc_deps;_install_pyfunc_deps(None, False)"'
        with tempfile.TemporaryDirectory() as tmp:
            docker_utils.generate_dockerfile(base_image=mlflow.models.docker_utils.UBUNTU_BASE_IMAGE, output_dir=tmp, entrypoint=sagemaker_image_entrypoint, env_manager=env_manager, mlflow_home=os.path.abspath(mlflow_home) if mlflow_home else None, model_install_steps=setup_container, disable_env_creation_at_runtime=False)
            docker_utils.build_image_from_context(tmp, image_name=container)
    if push:
        mlflow.sagemaker.push_image_to_ecr(container)