import warnings
import numpy
from sacred.dependencies import get_digest
from sacred.observers import RunObserver
import wandb
def artifact_event(self, name, filename, metadata=None, content_type=None):
    if content_type is None:
        content_type = 'file'
    artifact = wandb.Artifact(name, type=content_type)
    artifact.add_file(filename)
    self.run.log_artifact(artifact)