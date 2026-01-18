import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional
from wasabi import msg
from ..util import get_hash, join_command, load_project_config, run_command, working_dir
from .main import COMMAND, NAME, PROJECT_FILE, Arg, Opt, app
def check_workflows(workflows: List[str], workflow: Optional[str]=None) -> None:
    """Validate workflows provided in project.yml and check that a given
    workflow can be used to generate a DVC config.

    workflows (List[str]): Names of the available workflows.
    workflow (Optional[str]): The name of the workflow to convert.
    """
    if not workflows:
        msg.fail(f'No workflows defined in {PROJECT_FILE}. To generate a DVC config, define at least one list of commands.', exits=1)
    if workflow is not None and workflow not in workflows:
        msg.fail(f"Workflow '{workflow}' not defined in {PROJECT_FILE}. Available workflows: {', '.join(workflows)}", exits=1)
    if not workflow:
        msg.warn(f"No workflow specified for DVC pipeline. Using the first workflow defined in {PROJECT_FILE}: '{workflows[0]}'")