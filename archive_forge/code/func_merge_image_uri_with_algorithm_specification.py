import asyncio
import logging
from typing import Any, Dict, List, Optional, cast
import wandb
from wandb.apis.internal import Api
from wandb.sdk.launch.environment.aws_environment import AwsEnvironment
from wandb.sdk.launch.errors import LaunchError
from .._project_spec import EntryPoint, LaunchProject, get_entry_point_command
from ..builder.build import get_env_vars_dict
from ..registry.abstract import AbstractRegistry
from ..utils import (
from .abstract import AbstractRun, AbstractRunner, Status
def merge_image_uri_with_algorithm_specification(algorithm_specification: Optional[Dict[str, Any]], image_uri: Optional[str], entrypoint_command: List[str], args: Optional[List[str]]) -> Dict[str, Any]:
    """Create an AWS AlgorithmSpecification.

    AWS Sagemaker algorithms require a training image and an input mode. If the user
    does not specify the specification themselves, define the spec minimally using these
    two fields. Otherwise, if they specify the AlgorithmSpecification set the training
    image if it is not set.
    """
    if algorithm_specification is None:
        algorithm_specification = {'TrainingImage': image_uri, 'TrainingInputMode': 'File'}
    elif image_uri:
        algorithm_specification['TrainingImage'] = image_uri
    if entrypoint_command:
        algorithm_specification['ContainerEntrypoint'] = entrypoint_command
    if args:
        algorithm_specification['ContainerArguments'] = args
    if algorithm_specification['TrainingImage'] is None:
        raise LaunchError('Failed determine tag for training image')
    return algorithm_specification