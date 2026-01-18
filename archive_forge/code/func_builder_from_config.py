from typing import Any, Dict, Optional
import wandb
from wandb.apis.internal import Api
from wandb.docker import is_docker_installed
from wandb.sdk.launch.errors import LaunchError
from .builder.abstract import AbstractBuilder
from .environment.abstract import AbstractEnvironment
from .registry.abstract import AbstractRegistry
from .runner.abstract import AbstractRunner
def builder_from_config(config: Optional[Dict[str, Any]], environment: AbstractEnvironment, registry: AbstractRegistry) -> AbstractBuilder:
    """Create a builder from a config.

    This helper function is used to create a builder from a config. The
    config should have a "type" key that specifies the type of builder to import
    and create. The remaining keys are passed to the builder's from_config
    method. If the config is None or empty, a default builder is returned.

    The default builder will be a DockerBuilder if we find a working docker cli
    on the system, otherwise it will be a NoOpBuilder.

    Arguments:
        config (Dict[str, Any]): The builder config.
        registry (Registry): The registry of the builder.

    Returns:
        The builder.

    Raises:
        LaunchError: If the builder is not configured correctly.
    """
    if not config:
        if is_docker_installed():
            from .builder.docker_builder import DockerBuilder
            return DockerBuilder.from_config({}, environment, registry)
        from .builder.noop import NoOpBuilder
        return NoOpBuilder.from_config({}, environment, registry)
    builder_type = config.get('type')
    if builder_type is None:
        raise LaunchError('Could not create builder from config. Builder type not specified')
    if builder_type == 'docker':
        from .builder.docker_builder import DockerBuilder
        return DockerBuilder.from_config(config, environment, registry)
    if builder_type == 'kaniko':
        from .builder.kaniko_builder import KanikoBuilder
        return KanikoBuilder.from_config(config, environment, registry)
    if builder_type == 'noop':
        from .builder.noop import NoOpBuilder
        return NoOpBuilder.from_config(config, environment, registry)
    raise LaunchError(f'Could not create builder from config. Invalid builder type: {builder_type}')