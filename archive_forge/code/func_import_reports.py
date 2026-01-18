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
def import_reports(self, *, namespaces: Optional[Iterable[Namespace]]=None, limit: Optional[int]=None, remapping: Optional[Dict[Namespace, Namespace]]=None):
    logger.info('START: Importing reports')
    logger.info('Collecting reports')
    reports = self._collect_reports(namespaces=namespaces, limit=limit)
    logger.info('Importing reports')

    def _import_report_wrapped(report):
        namespace = Namespace(report.entity, report.project)
        if remapping is not None and namespace in remapping:
            namespace = remapping[namespace]
        logger.debug(f'Importing report={report!r}, namespace={namespace!r}')
        self._import_report(report, namespace=namespace)
        logger.debug(f'Finished importing report={report!r}, namespace={namespace!r}')
    for_each(_import_report_wrapped, reports)
    logger.info('END: Importing reports')