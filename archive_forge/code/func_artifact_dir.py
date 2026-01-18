import os
import shutil
from mlflow.store.artifact.artifact_repo import ArtifactRepository, verify_artifact_path
from mlflow.utils.file_utils import (
from mlflow.utils.uri import validate_path_is_safe
@property
def artifact_dir(self):
    return self._artifact_dir