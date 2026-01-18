import logging
import warnings
import entrypoints
from mlflow.tracking.context.databricks_cluster_context import DatabricksClusterRunContext
from mlflow.tracking.context.databricks_command_context import DatabricksCommandRunContext
from mlflow.tracking.context.databricks_job_context import DatabricksJobRunContext
from mlflow.tracking.context.databricks_notebook_context import DatabricksNotebookRunContext
from mlflow.tracking.context.databricks_repo_context import DatabricksRepoRunContext
from mlflow.tracking.context.default_context import DefaultRunContext
from mlflow.tracking.context.git_context import GitRunContext
from mlflow.tracking.context.system_environment_context import SystemEnvironmentContext
def resolve_tags(tags=None):
    """Generate a set of tags for the current run context. Tags are resolved in the order,
    contexts are registered. Argument tags are applied last.

    This function iterates through all run context providers in the registry. Additional context
    providers can be registered as described in
    :py:class:`mlflow.tracking.context.RunContextProvider`.

    Args:
        tags: A dictionary of tags to override. If specified, tags passed in this argument will
            override those inferred from the context.

    Returns:
        A dictionary of resolved tags.
    """
    all_tags = {}
    for provider in _run_context_provider_registry:
        try:
            if provider.in_context():
                all_tags.update(provider.tags())
        except Exception as e:
            _logger.warning('Encountered unexpected error during resolving tags: %s', e)
    if tags is not None:
        all_tags.update(tags)
    return all_tags