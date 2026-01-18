import warnings
import numpy
from sacred.dependencies import get_digest
from sacred.observers import RunObserver
import wandb
def completed_event(self, stop_time, result):
    if result:
        if not isinstance(result, tuple):
            result = (result,)
        for i, r in enumerate(result):
            if isinstance(r, float) or isinstance(r, int):
                wandb.log({f'result_{i}': float(r)})
            elif isinstance(r, dict):
                wandb.log(r)
            elif isinstance(r, object):
                artifact = wandb.Artifact(f'result_{i}.pkl', type='result')
                artifact.add_file(r)
                self.run.log_artifact(artifact)
            elif isinstance(r, numpy.ndarray):
                wandb.log({f'result_{i}': wandb.Image(r)})
            else:
                warnings.warn(f"logging results does not support type '{type(r)}' results. Ignoring this result", stacklevel=2)