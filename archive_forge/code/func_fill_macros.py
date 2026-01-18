import enum
import logging
import os
import tempfile
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Dict, List, Optional, cast
import wandb
import wandb.docker as docker
from wandb.apis.internal import Api
from wandb.errors import CommError
from wandb.sdk.launch import utils
from wandb.sdk.lib.runid import generate_id
from .errors import LaunchError
from .utils import LOG_PREFIX, recursive_macro_sub
def fill_macros(self, image: str) -> Dict[str, Any]:
    """Substitute values for macros in resource arguments.

        Certain macros can be used in resource args. These macros allow the
        user to set resource args dynamically in the context of the
        run being launched. The macros are given in the ${macro} format. The
        following macros are currently supported:

        ${project_name} - the name of the project the run is being launched to.
        ${entity_name} - the owner of the project the run being launched to.
        ${run_id} - the id of the run being launched.
        ${run_name} - the name of the run that is launching.
        ${image_uri} - the URI of the container image for this run.

        Additionally, you may use ${<ENV-VAR-NAME>} to refer to the value of any
        environment variables that you plan to set in the environment of any
        agents that will receive these resource args.

        Calling this method will overwrite the contents of self.resource_args
        with the substituted values.

        Args:
            image (str): The image name to fill in for ${wandb-image}.

        Returns:
            None
        """
    update_dict = {'project_name': self.target_project, 'entity_name': self.target_entity, 'run_id': self.run_id, 'run_name': self.name, 'image_uri': image, 'author': self.author}
    update_dict.update(os.environ)
    result = recursive_macro_sub(self.resource_args, update_dict)
    return cast(Dict[str, Any], result)