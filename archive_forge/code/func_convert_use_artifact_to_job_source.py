import json
import logging
import os
import re
import sys
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
import wandb
from wandb.sdk.artifacts.artifact import Artifact
from wandb.sdk.data_types._dtypes import TypeRegistry
from wandb.sdk.lib.filenames import DIFF_FNAME, METADATA_FNAME, REQUIREMENTS_FNAME
from wandb.util import make_artifact_name_safe
from .settings_static import SettingsStatic
def convert_use_artifact_to_job_source(use_artifact: 'UseArtifactRecord') -> PartialJobSourceDict:
    source_info = use_artifact.partial.source_info
    source_info_dict: JobSourceDict = {'_version': 'v0', 'source_type': source_info.source_type, 'runtime': source_info.runtime}
    if source_info.source_type == 'repo':
        entrypoint = [str(x) for x in source_info.source.git.entrypoint]
        git_source: GitSourceDict = {'git': {'remote': source_info.source.git.git_info.remote, 'commit': source_info.source.git.git_info.commit}, 'entrypoint': entrypoint, 'notebook': source_info.source.git.notebook}
        source_info_dict.update({'source': git_source})
    elif source_info.source_type == 'artifact':
        entrypoint = [str(x) for x in source_info.source.artifact.entrypoint]
        artifact_source: ArtifactSourceDict = {'artifact': source_info.source.artifact.artifact, 'entrypoint': entrypoint, 'notebook': source_info.source.artifact.notebook}
        source_info_dict.update({'source': artifact_source})
    elif source_info.source_type == 'image':
        image_source: ImageSourceDict = {'image': source_info.source.image.image}
        source_info_dict.update({'source': image_source})
    partal_job_source_dict: PartialJobSourceDict = {'job_name': use_artifact.partial.job_name.split(':')[0], 'job_source_info': source_info_dict}
    return partal_job_source_dict