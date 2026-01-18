import concurrent.futures
import logging
import os
import queue
import tempfile
import threading
import time
from typing import TYPE_CHECKING, Optional, Tuple
import wandb
import wandb.util
from wandb.filesync import stats, step_checksum, step_upload
from wandb.sdk.lib.paths import LogicalPath
def store_manifest_files(self, manifest: 'ArtifactManifest', artifact_id: str, save_fn: 'SaveFn', save_fn_async: 'SaveFnAsync') -> None:
    event = step_checksum.RequestStoreManifestFiles(manifest, artifact_id, save_fn, save_fn_async)
    self._incoming_queue.put(event)