import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional
from wasabi import msg
from ..util import get_hash, join_command, load_project_config, run_command, working_dir
from .main import COMMAND, NAME, PROJECT_FILE, Arg, Opt, app
def ensure_dvc(project_dir: Path) -> None:
    """Ensure that the "dvc" command is available and that the current project
    directory is an initialized DVC project.
    """
    try:
        subprocess.run(['dvc', '--version'], stdout=subprocess.DEVNULL)
    except Exception:
        msg.fail("To use Weasel with DVC (Data Version Control), DVC needs to be installed and the 'dvc' command needs to be available", 'You can install the Python package from pip (pip install dvc) or conda (conda install -c conda-forge dvc). For more details, see the documentation: https://dvc.org/doc/install', exits=1)
    if not (project_dir / '.dvc').exists():
        msg.fail('Project not initialized as a DVC project', "To initialize a DVC project, you can run 'dvc init' in the project directory. For more details, see the documentation: https://dvc.org/doc/command-reference/init", exits=1)