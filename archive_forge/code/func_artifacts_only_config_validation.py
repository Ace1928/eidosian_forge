import click
from mlflow.store.tracking import DEFAULT_ARTIFACTS_URI, DEFAULT_LOCAL_FILE_AND_ARTIFACT_PATH
from mlflow.utils.logging_utils import eprint
from mlflow.utils.uri import is_local_uri
def artifacts_only_config_validation(artifacts_only: bool, backend_store_uri: str) -> None:
    if artifacts_only and (not _is_default_backend_store_uri(backend_store_uri)):
        msg = f"You are starting a tracking server in `--artifacts-only` mode and have provided a value for `--backend_store_uri`: '{backend_store_uri}'. A tracking server in `--artifacts-only` mode cannot have a value set for `--backend_store_uri` to properly proxy access to the artifact storage location."
        raise click.UsageError(message=msg)