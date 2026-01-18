import itertools
import json
import logging
import numbers
import os
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime as dt
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple
from unittest.mock import patch
import filelock
import polars as pl
import requests
import urllib3
import yaml
from wandb_gql import gql
import wandb
import wandb.apis.reports as wr
from wandb.apis.public import ArtifactCollection, Run
from wandb.apis.public.files import File
from wandb.apis.reports import Report
from wandb.util import coalesce, remove_keys_with_none_values
from . import validation
from .internals import internal
from .internals.protocols import PathStr, Policy
from .internals.util import Namespace, for_each
def import_artifact_sequences(self, *, namespaces: Optional[Iterable[Namespace]]=None, incremental: bool=True, max_workers: Optional[int]=None, remapping: Optional[Dict[Namespace, Namespace]]=None):
    """Import all artifact sequences from `namespaces`.

        Note: There is a known bug with the AWS backend where artifacts > 2048MB will fail to upload.  This seems to be related to multipart uploads, but we don't have a fix yet.
        """
    logger.info('START: Importing artifact sequences')
    _clear_fname(ARTIFACT_ERRORS_FNAME)
    logger.info('Collecting artifact sequences')
    seqs = list(self._collect_artifact_sequences(namespaces=namespaces))
    logger.info('Validating artifact sequences')
    self._validate_artifact_sequences(seqs, incremental=incremental, remapping=remapping)
    logger.info('Collecting failed artifact sequences')
    seqs = list(self._collect_failed_artifact_sequences())
    logger.info(f'Importing artifact sequences, len(seqs)={len(seqs)!r}')

    def _import_artifact_sequence_wrapped(seq):
        namespace = Namespace(seq.entity, seq.project)
        if remapping is not None and namespace in remapping:
            namespace = remapping[namespace]
        logger.debug(f'Importing artifact sequence seq={seq!r}, namespace={namespace!r}')
        self._import_artifact_sequence(seq, namespace=namespace)
        logger.debug(f'Finished importing artifact sequence seq={seq!r}, namespace={namespace!r}')
    for_each(_import_artifact_sequence_wrapped, seqs, max_workers=max_workers)
    logger.debug(f'Using artifact sequences, len(seqs)={len(seqs)!r}')

    def _use_artifact_sequence_wrapped(seq):
        namespace = Namespace(seq.entity, seq.project)
        if remapping is not None and namespace in remapping:
            namespace = remapping[namespace]
        logger.debug(f'Using artifact sequence seq={seq!r}, namespace={namespace!r}')
        self._use_artifact_sequence(seq, namespace=namespace)
        logger.debug(f'Finished using artifact sequence seq={seq!r}, namespace={namespace!r}')
    for_each(_use_artifact_sequence_wrapped, seqs, max_workers=max_workers)
    logger.info('Cleaning up dummy runs')
    self._cleanup_dummy_runs(namespaces=namespaces, remapping=remapping)
    logger.info('END: Importing artifact sequences')