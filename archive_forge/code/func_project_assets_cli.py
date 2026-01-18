import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Optional
import requests
import typer
from wasabi import msg
from ..util import SimpleFrozenDict, download_file, ensure_path, get_checksum
from ..util import get_git_version, git_checkout, load_project_config
from ..util import parse_config_overrides, working_dir
from .main import PROJECT_FILE, Arg, Opt, app
@app.command('assets', context_settings={'allow_extra_args': True, 'ignore_unknown_options': True})
def project_assets_cli(ctx: typer.Context, project_dir: Path=Arg(Path.cwd(), help='Path to cloned project. Defaults to current working directory.', exists=True, file_okay=False), sparse_checkout: bool=Opt(False, '--sparse', '-S', help='Use sparse checkout for assets provided via Git, to only check out and clone the files needed. Requires Git v22.2+.'), extra: bool=Opt(False, '--extra', '-e', help="Download all assets, including those marked as 'extra'.")):
    """Fetch project assets like datasets and pretrained weights. Assets are
    defined in the "assets" section of the project.yml. If a checksum is
    provided in the project.yml, the file is only downloaded if no local file
    with the same checksum exists.

    DOCS: https://github.com/explosion/weasel/tree/main/docs/tutorial/directory-and-assets.md
    """
    overrides = parse_config_overrides(ctx.args)
    project_assets(project_dir, overrides=overrides, sparse_checkout=sparse_checkout, extra=extra)